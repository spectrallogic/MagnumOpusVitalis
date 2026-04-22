"""
FlowRunner — multi-clock asyncio runner.

Four scheduled coroutines, all in one asyncio loop in a daemon thread:

    flow_clock        ~50ms   — substrate update (cheap math only)
    perception_clock  ~200ms  — pressure recompute, subjective time, decay
    expensive_clock   ~1.5s   — silent forward passes, knowledge sparks
    slow_clock        ~30s    — neuromod baseline shifts, false memory consolidation

Why threads + asyncio together:
- Flask handlers run in the main thread and call engine.converse().
- The flow has to keep ticking *during* generation (the bus is alive
  even mid-response). So flow lives in its own thread.
- Inside that thread, the four clocks share one asyncio loop so they
  interleave cooperatively without contending for a GIL-bound lock.
- The bus uses a threading.Lock internally so generation (main thread)
  can read/write safely while the flow thread is updating.
- Expensive-clock regions that touch the model grab `expensive_lock`
  (asyncio) so two model passes never overlap inside the flow thread.
  Generation grabs the bus lock; the model itself is single-threaded
  with PyTorch handling tensor ops.
"""

import asyncio
import threading
import time
from typing import List, Optional

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.config import ClockConfig
from magnum_opus_v2.region import Region


class FlowRunner:
    def __init__(
        self,
        bus: LatentBus,
        regions: List[Region],
        clock_config: Optional[ClockConfig] = None,
        neuromod: object = None,
        verbose_errors: bool = False,
    ):
        self.bus = bus
        self.regions = list(regions)
        self.clock_config = clock_config or ClockConfig()
        self.neuromod = neuromod
        self.verbose_errors = verbose_errors

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._ready_evt = threading.Event()

        # Per-clock metrics: tick count and last-observed wall-clock period.
        self.metrics = {
            name: {"ticks": 0, "last_wall_dt": 0.0}
            for name in ("flow", "perception", "expensive", "slow")
        }

    # ------------------------------------------------------------------
    # Region registry (mutable so we can add regions before/after start)
    # ------------------------------------------------------------------
    def add_region(self, region: Region) -> None:
        self.regions.append(region)

    def regions_on_clock(self, clock_name: str) -> List[Region]:
        return [r for r in self.regions if getattr(r, "clock", "flow") == clock_name]

    # ------------------------------------------------------------------
    # Clock implementations
    # ------------------------------------------------------------------
    async def _flow_clock(self) -> None:
        dt = self.clock_config.flow_dt_seconds
        last = time.monotonic()
        while not self._stop_evt.is_set():
            start = time.monotonic()
            try:
                self.bus.step(
                    self.regions_on_clock("flow"),
                    self.neuromod,
                    dt=dt,
                    verbose_errors=self.verbose_errors,
                )
            except Exception as e:  # noqa: BLE001
                if self.verbose_errors:
                    print(f"  [flow_clock error] {e}")
            self.metrics["flow"]["ticks"] += 1
            self.metrics["flow"]["last_wall_dt"] = start - last
            last = start
            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0.0, dt - elapsed))

    async def _generic_clock(self, clock_name: str, dt: float) -> None:
        # perception / expensive / slow: side-effect calls only. Returned
        # perturbations are injected into bus.velocity directly without
        # re-running the substrate integration step (only the flow clock
        # owns integration).
        last = time.monotonic()
        while not self._stop_evt.is_set():
            start = time.monotonic()
            regions = self.regions_on_clock(clock_name)
            try:
                self.bus.run_side_effects(
                    regions, self.neuromod, dt=dt,
                    verbose_errors=self.verbose_errors,
                )
            except Exception as e:  # noqa: BLE001
                if self.verbose_errors:
                    print(f"  [{clock_name}_clock error] {e}")
            self.metrics[clock_name]["ticks"] += 1
            self.metrics[clock_name]["last_wall_dt"] = start - last
            last = start
            elapsed = time.monotonic() - start
            await asyncio.sleep(max(0.0, dt - elapsed))

    async def _run(self) -> None:
        self._stop_evt = asyncio.Event()
        self._ready_evt.set()
        await asyncio.gather(
            self._flow_clock(),
            self._generic_clock("perception", self.clock_config.perception_dt_seconds),
            self._generic_clock("expensive", self.clock_config.expensive_dt_seconds),
            self._generic_clock("slow", self.clock_config.slow_dt_seconds),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the flow loop in a daemon thread. Returns once it's running."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._ready_evt.clear()

        def thread_main() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run())
            except Exception as e:  # noqa: BLE001
                if self.verbose_errors:
                    print(f"  [flow thread crashed] {e}")
            finally:
                try:
                    self._loop.close()
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(
            target=thread_main, name="v2-flow", daemon=True
        )
        self._thread.start()
        self._ready_evt.wait(timeout=2.0)

    def stop(self, timeout: float = 2.0) -> None:
        if self._loop is None or self._stop_evt is None:
            return
        self._loop.call_soon_threadsafe(self._stop_evt.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

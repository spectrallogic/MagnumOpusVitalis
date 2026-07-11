"""
Steering hook + bus-driven driver.

The hook attaches via register_forward_hook on a target transformer layer
(GPT-2 / LLaMA / Mistral compatible). When active, it adds a steering
vector to the layer's hidden states during the forward pass.

Two ways to drive it:

  set_steering(vec)      — static vector, applied identically to every
                           forward pass until cleared. Used for one-shot
                           silent passes (DefaultMode, SpeculativeFutures).

  set_provider(callable) — LIVE mode: the callable is invoked on EVERY
                           forward pass and returns the vector to inject
                           (or None). Generation passes the driver's read()
                           here, so each token is steered by the bus state
                           *at that moment* — emotions shift mid-sentence.
                           This is what makes the stream realtime.

Hidden-state capture is opt-in (`capture_enabled`) — regions that need to
observe the model (DefaultMode, SpeculativeFutures) turn it on around their
silent pass. Generation leaves it off, so long conversations don't
accumulate a tensor per generated token.
"""

from typing import Callable, List, Optional

import torch

from magnum_opus_v2.bus import LatentBus


class SteeringHook:
    """Forward-hook-based latent-space intervention.

    Captures hidden states (when enabled) and, when active, adds a steering
    vector to the layer's output. Handles three output shapes:
        GPT-2:          tuple (hidden_states, presents, ...)
        LLaMA/Mistral:  plain Tensor or tuple
        HF ModelOutput: dataclass — first field replaced

    Attaches to the PRIMARY layer at full strength and optionally to its
    neighbors at reduced strength (`layer_span`) — the mind whispers into
    several depths, not one. Capture and feedback fire only on the primary.

    `feedback` closes the traffic loop: every `feedback_every` primary
    passes, the current last-token hidden state (pre-steering — the model's
    own thought, not our injection) is handed to the callback, which lets
    generation write back into the bus while speaking.
    """

    def __init__(self):
        self.steering_vector: Optional[torch.Tensor] = None
        self.provider: Optional[Callable[[], Optional[torch.Tensor]]] = None
        self.captured_states: List[torch.Tensor] = []
        self.capture_enabled = False
        self.active = False
        self.feedback: Optional[Callable[[torch.Tensor], None]] = None
        self.feedback_every: int = 8
        self.feedback_enabled = True   # generate_raw() and external
        self._fb_count = 0             # measurement passes disable this
        self._handles: List = []       # so they never perturb the bus

    def set_steering(self, vector: Optional[torch.Tensor]) -> None:
        """Static steering vector. Clears any live provider."""
        self.steering_vector = vector
        self.provider = None
        self.active = vector is not None

    def set_provider(
        self, provider: Optional[Callable[[], Optional[torch.Tensor]]]
    ) -> None:
        """Live steering: `provider()` is called on every forward pass."""
        self.provider = provider
        self.steering_vector = None
        self.active = provider is not None

    def set_feedback(
        self, fn: Optional[Callable[[torch.Tensor], None]], every: int = 8,
    ) -> None:
        """Thought→feeling tap. `fn(hidden_vec)` is called with the primary
        layer's last-token hidden state every `every` passes."""
        self.feedback = fn
        self.feedback_every = max(1, int(every))
        self._fb_count = 0

    def _current_vector(self) -> Optional[torch.Tensor]:
        if self.provider is not None:
            try:
                return self.provider()
            except Exception:  # noqa: BLE001 — never crash a forward pass
                return None
        return self.steering_vector

    def _make_hook_fn(self, scale: float, primary: bool):
        def hook_fn(module, input, output):
            is_tensor = isinstance(output, torch.Tensor)
            is_tuple = isinstance(output, tuple)
            hidden_states = output if is_tensor else output[0]

            if primary and self.capture_enabled:
                self.captured_states.append(hidden_states.detach().clone())

            if primary and self.feedback is not None and self.feedback_enabled:
                self._fb_count += 1
                if self._fb_count % self.feedback_every == 0:
                    try:
                        self.feedback(hidden_states[0, -1].detach())
                    except Exception:  # noqa: BLE001
                        pass

            if self.active:
                sv = self._current_vector()
                if sv is not None:
                    sv = sv.to(hidden_states.device).to(hidden_states.dtype)
                    modified = hidden_states + (sv * scale).unsqueeze(0).unsqueeze(0)
                    if is_tensor:
                        return modified
                    elif is_tuple:
                        return (modified,) + output[1:]
                    else:
                        keys = list(output.keys())
                        output[keys[0]] = modified
                        return output
            return output
        return hook_fn

    # Kept for backward compatibility (primary-layer semantics).
    def hook_fn(self, module, input, output):
        return self._make_hook_fn(1.0, True)(module, input, output)

    def clear(self) -> None:
        self.captured_states = []

    @staticmethod
    def _layer_list(model):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise ValueError(
            f"Unknown model architecture. Model type: {type(model).__name__}"
        )

    def attach(
        self, model, target_layer: int,
        layer_span: int = 1, span_scale: float = 0.4,
    ) -> "SteeringHook":
        """Attach to target_layer (full strength) and ±layer_span neighbors
        (span_scale strength). span=0 restores single-layer behavior."""
        layers = self._layer_list(model)
        n = len(layers)
        self._handles.append(
            layers[target_layer].register_forward_hook(
                self._make_hook_fn(1.0, primary=True))
        )
        for off in range(1, max(0, int(layer_span)) + 1):
            for idx in (target_layer - off, target_layer + off):
                if 0 <= idx < n and idx != target_layer:
                    self._handles.append(
                        layers[idx].register_forward_hook(
                            self._make_hook_fn(float(span_scale), primary=False))
                    )
        return self

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []


class BusSteeringDriver:
    """
    Reads bus.state and produces the steering vector the hook injects:
    steering_vector = bus.state * steering_strength, raw and unsmoothed.
    Since the bus lives in the model's hidden_dim space and is shaped by
    regions that already emit perturbations along emotion-vector
    directions, the bus state IS already a meaningful steering vector.

    (An EMA-smoothing option used to live here; no live path ever set it
    above zero, so it was dead mechanism and the Era-4 audit removed it.
    The bus's own OU dynamics are the smoothing.)
    """

    def __init__(self, bus: LatentBus, steering_strength: float = 1.0):
        self.bus = bus
        self.steering_strength = float(steering_strength)

    def read(self) -> torch.Tensor:
        return self.bus.state.detach().clone() * self.steering_strength

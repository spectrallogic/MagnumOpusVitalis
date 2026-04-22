"""
V2Engine — thin orchestrator.

Assembles: LatentBus + NeuromodState + all regions + SteeringHook + driver +
FlowRunner. Provides a small public surface:

    engine = V2Engine.from_profile(model, tokenizer, profile, device=...)
    engine.start()                       # spin up flow runner
    engine.user_message("hi")            # mark interaction, light keyword stim
    text = engine.converse("hello")      # generate with live bus driving steering
    snap = engine.snapshot()             # everything at-a-glance
    engine.stop()

The engine itself does NOT orchestrate per-tick logic — that's the bus +
clocks. This class just wires regions together at construction and exposes
external entry points (user message, generate, snapshot).
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from magnum_opus.profile import ModelProfile

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.config import V2Config
from magnum_opus_v2.flow import FlowRunner
from magnum_opus_v2.neuromod import NeuromodState, NeuromodulatorRegion
from magnum_opus_v2.steering_hook import SteeringHook, BusSteeringDriver
from magnum_opus_v2.regions import (
    Limbic, Temporal, SubconsciousStack, Memory, FalseMemoryConfabulator,
    Salience, Executive, DefaultMode, KnowledgeSparks,
    NoiseSampler, TokenEmbeddingSampler,
)


# Coarse keyword detection — same idea as v1's detect_emotional_content.
# Only used to seed Limbic stimulation when a user message arrives. Real
# steering still comes from the bus.
_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "joy":      ["happy", "great", "awesome", "love", "wonderful", "excited"],
    "sadness":  ["sad", "lonely", "unhappy", "down", "sorry", "miss"],
    "fear":     ["afraid", "scared", "worried", "anxious", "nervous"],
    "anger":    ["angry", "mad", "furious", "hate", "annoyed"],
    "disgust":  ["disgusting", "gross", "yuck", "awful"],
    "surprise": ["wow", "really?", "what?!", "no way"],
    "trust":    ["trust", "believe", "rely", "honest"],
    "curious":  ["curious", "wonder", "why", "how"],
    "calm":     ["calm", "peaceful", "relax", "fine"],
    "desperate":["please", "need", "help", "urgent"],
}


def _detect_emotions(text: str) -> Dict[str, float]:
    text_l = text.lower()
    out: Dict[str, float] = {}
    for emo, words in _EMOTION_KEYWORDS.items():
        score = sum(1 for w in words if w in text_l)
        if score > 0:
            out[emo] = min(1.0, 0.4 + 0.2 * score)
    return out


@dataclass
class V2Engine:
    bus: LatentBus
    neuromod: NeuromodState
    flow: FlowRunner
    hook: SteeringHook
    driver: BusSteeringDriver
    limbic: Limbic
    temporal: Temporal
    subc: SubconsciousStack
    memory: Memory
    confab: FalseMemoryConfabulator
    salience: Salience
    executive: Executive
    default_mode: Optional[DefaultMode]
    sparks: Optional[KnowledgeSparks]
    model: Any
    tokenizer: Any
    device: str = "cpu"
    model_lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_profile(
        cls,
        model,
        tokenizer,
        profile: ModelProfile,
        device: str = "cpu",
        config: Optional[V2Config] = None,
        # Behavior switches
        enable_default_mode: bool = True,
        enable_knowledge_sparks: bool = True,
        # Speech callback
        on_should_speak=None,
    ) -> "V2Engine":
        cfg = config or V2Config(hidden_dim=profile.hidden_dim, device=device)

        bus = LatentBus(profile.hidden_dim, device=device, config=cfg.bus)
        neuromod = NeuromodState()
        model_lock = threading.Lock()

        # Limbic + bus baseline
        limbic = Limbic(profile.vectors, device=device, steering_strength=1.0)
        bus.set_baseline(limbic.baseline_vector(), weight=1.0)

        temporal = Temporal(profile.vectors, device=device, steering_strength=0.3)

        # Memory
        memory = Memory(device=device, capacity=200)
        confab = FalseMemoryConfabulator(memory, per_tick_count=1)

        # SubconsciousStack with three samplers: noise, token-embedding, memory
        embedding_matrix = model.get_input_embeddings().weight.detach().to(device)
        forbidden = [
            tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id,
        ]
        forbidden = [f for f in forbidden if f is not None]
        samplers = [
            NoiseSampler(profile.hidden_dim, device=device, magnitude=1.0),
            TokenEmbeddingSampler(
                embedding_matrix, device=device,
                candidate_pool=64, forbidden_token_ids=forbidden,
            ),
            memory.make_sampler(),
        ]
        subc = SubconsciousStack(
            hidden_dim=profile.hidden_dim, device=device, samplers=samplers,
            l0_per_sampler_count=4, l0_emotion_bias_strength=0.5,
            l1_keep_top_k=8, l2_keep_top_k=3,
            l2_surprise_probability=0.15,
            l3_perturbation_strength=0.4,
            emotion_vectors=profile.vectors,
        )

        salience = Salience(subc, history_size=20)
        executive = Executive(
            base_threshold=0.55,
            pressure_growth=0.40,
            pressure_decay=0.985,
            on_should_speak=on_should_speak,
        )

        # Steering hook + driver
        hook = SteeringHook()
        hook.attach(model, profile.target_layer)
        driver = BusSteeringDriver(bus, steering_strength=1.0, smoothing=0.0)

        # Optional model-touching regions
        default_mode = None
        if enable_default_mode:
            default_mode = DefaultMode(
                model=model, tokenizer=tokenizer,
                hook=hook, driver=driver,
                limbic=limbic,
                emotion_vectors=profile.vectors,
                device=device,
                idle_drift_amplitude=0.04,
                model_lock=model_lock,
            )

        sparks = None
        sparks_fire = None
        if enable_knowledge_sparks:
            sparks = KnowledgeSparks(
                model=model, limbic=limbic, device=device,
                fire_probability=0.05,
                spark_strength=0.6,
            )
            sparks_fire = sparks.fire_companion()

        # Neuromodulator slow-clock region
        neuromod_region = NeuromodulatorRegion(
            neuromod=neuromod,
            limbic_provider=limbic.snapshot,
            executive_provider=executive.snapshot,
        )

        regions = [
            limbic, temporal, subc, salience, executive,
            memory, confab, neuromod_region,
        ]
        if default_mode is not None:
            regions.append(default_mode)
        if sparks is not None:
            regions.append(sparks)
            regions.append(sparks_fire)

        flow = FlowRunner(
            bus=bus, regions=regions,
            clock_config=cfg.clock,
            neuromod=neuromod,
            verbose_errors=True,
        )

        return cls(
            bus=bus, neuromod=neuromod, flow=flow,
            hook=hook, driver=driver,
            limbic=limbic, temporal=temporal, subc=subc,
            memory=memory, confab=confab,
            salience=salience, executive=executive,
            default_mode=default_mode, sparks=sparks,
            model=model, tokenizer=tokenizer, device=device,
            model_lock=model_lock,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.flow.start()

    def stop(self) -> None:
        self.flow.stop()

    # ------------------------------------------------------------------
    # External entry points
    # ------------------------------------------------------------------
    def user_message(self, text: str) -> None:
        """User just sent something. Stimulate Limbic from keywords, mark
        Temporal + Executive interaction, push the live emotion blend into
        the subconscious so its filtering is informed."""
        stim = _detect_emotions(text)
        if stim:
            self.limbic.stimulate_many(stim, neuromod=self.neuromod)
        self.temporal.mark_interaction(self.bus)
        self.executive.mark_interaction()
        # Also force a memory capture at the moment of interaction
        self.memory.force_capture(
            self.bus, importance=1.0, tag=f"user_msg:{text[:40]}",
        )
        self.subc.set_emotion_blend(self.limbic.snapshot()["blended"])

    def converse(
        self, prompt: str, max_new_tokens: int = 50,
        do_sample: bool = True, top_p: float = 0.92, temperature: float = 0.9,
        seed: Optional[int] = None,
    ) -> str:
        """Generate with live bus driving the steering hook."""
        self.user_message(prompt)
        if seed is not None:
            torch.manual_seed(seed)

        # Push current emotion into subconscious so steering reflects it.
        self.subc.set_emotion_blend(self.limbic.snapshot()["blended"])

        with self.model_lock:
            self.hook.set_steering(self.driver.read())
            try:
                enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **enc, max_new_tokens=max_new_tokens,
                        do_sample=do_sample, top_p=top_p, temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            finally:
                self.hook.set_steering(None)

        # Capture a memory of the response moment too
        self.memory.force_capture(
            self.bus, importance=1.2, tag=f"response:{text[len(prompt):][:40]}",
        )
        # Mark spoke (resets executive pressure even for non-autonomous turns)
        self.executive.mark_spoke()
        return text

    def speak_autonomously(
        self, max_new_tokens: int = 30,
        do_sample: bool = True, top_p: float = 0.92, temperature: float = 1.0,
    ) -> str:
        """Called from on_should_speak callback. Generate from BOS with the
        current bus steering, no user prompt."""
        bos = (
            self.tokenizer.bos_token_id
            or self.tokenizer.eos_token_id
            or 0
        )
        with self.model_lock:
            self.hook.set_steering(self.driver.read())
            try:
                seed_t = torch.tensor([[bos]], device=self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        seed_t, max_new_tokens=max_new_tokens,
                        do_sample=do_sample, top_p=top_p, temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            finally:
                self.hook.set_steering(None)

        self.executive.mark_spoke()
        return text

    # ------------------------------------------------------------------
    # Snapshot — everything the dashboard wants
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "bus":          self.bus.snapshot(),
            "neuromod":     self.neuromod.snapshot(),
            "limbic":       self.limbic.snapshot(),
            "temporal":     self.temporal.snapshot(self.bus),
            "subconscious": self.subc.snapshot(),
            "memory":       self.memory.snapshot(),
            "salience":     self.salience.snapshot(),
            "executive":    self.executive.snapshot(),
            "default_mode": self.default_mode.snapshot() if self.default_mode else None,
            "sparks":       self.sparks.snapshot() if self.sparks else None,
            "flow_metrics": self.flow.metrics,
        }

"""Offline causal and architectural checks on actual tiny transformer blocks."""

from dataclasses import asdict

import pytest
import torch
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

from magnum_opus_v2.bus import LatentBus
from magnum_opus_v2.extraction import extract_vectors, read_block_output
from magnum_opus_v2.imagination import search_futures
from magnum_opus_v2.profile import Baseline, ModelProfile, ProfileMetadata
from magnum_opus_v2.regions.speculative import SpeculativeFutures
from magnum_opus_v2.regions.subconscious import SubconsciousStack
from magnum_opus_v2.steering_hook import SteeringHook


class TinyTokenizer:
    all_special_ids = []
    bos_token_id = eos_token_id = 0
    pad_token_id = 0
    chat_template = None

    def encode(self, text, **kwargs):
        return [ord(c) % 30 + 1 for c in text]

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)[:kwargs.get("max_length", 128)]
        if return_tensors:
            return BatchEncoding({"input_ids": torch.tensor([ids]),
                                  "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture
def tiny_model():
    torch.manual_seed(4)
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=128,
                                    n_embd=16, n_layer=3, n_head=2,
                                    resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()


def make_profile():
    return ModelProfile(ProfileMetadata("tiny", 1, 16, 3, ["calm"], "test"),
                        {"calm": torch.eye(16)[0]}, Baseline({"calm": 0}, 1))


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_extraction_and_intervention_use_exact_same_block(tiny_model, layer):
    enc = TinyTokenizer()("check", return_tensors="pt")
    hook = SteeringHook().attach(tiny_model, layer)
    with hook.isolated(capture=True):
        tiny_model(**enc)
        expected = hook.captured_states[0].mean(1).squeeze(0)
        measured = read_block_output(tiny_model, enc, layer)
    assert torch.allclose(measured, expected)
    assert len(hook._handles) == 1  # one profile boundary, one intervention site
    hook.detach()


def test_failed_measurement_restores_hook_and_removes_temporary_capture(tiny_model):
    hook = SteeringHook().attach(tiny_model, 1)
    provider = lambda: torch.ones(16)
    hook.set_provider(provider)
    hook.set_feedback(lambda h: pytest.fail("measurement leaked into speech feedback"), every=1)
    prior = hook.captured_states
    count = len(tiny_model.transformer.h[1]._forward_hooks)
    with pytest.raises(RuntimeError):
        with hook.isolated(torch.zeros(16), capture=True):
            tiny_model(torch.tensor([[1, 2]]))
            raise RuntimeError("interrupted")
    assert hook.provider is provider and hook.active and hook.feedback_enabled
    assert hook.captured_states is prior and hook._fb_count == 0
    with pytest.raises((IndexError, RuntimeError)):
        read_block_output(tiny_model, {"input_ids": torch.tensor([[1000]])}, 1)
    assert len(tiny_model.transformer.h[1]._forward_hooks) == count
    hook.detach()


def test_extraction_rejects_zero_contrast_and_masks_padding(tiny_model):
    tok = TinyTokenizer()
    with pytest.raises(ValueError, match="nonzero"):
        extract_vectors(tiny_model, tok, 1, "cpu", temporal_pairs={}, verbose=False,
                        emotion_pairs={"same": {"positive": ["abc"], "negative": ["abc"]}})
    plain = read_block_output(tiny_model, tok("ab", return_tensors="pt"), 1)
    padded = read_block_output(tiny_model, {"input_ids": torch.tensor([[8, 9, 0, 0]]),
                                         "attention_mask": torch.tensor([[1, 1, 0, 0]])}, 1)
    assert torch.allclose(plain, padded, atol=1e-6)


def test_legacy_profile_is_identified_and_rejected():
    profile = make_profile()
    old = asdict(profile.metadata)
    old.pop("activation_site")
    profile.metadata = ProfileMetadata.from_dict(old)
    with pytest.raises(ValueError, match="Re-extract"):
        profile.validate_activation_site()


def test_recursive_outcome_can_overturn_first_impression():
    seen = []

    def rollout(seed, parent, deadline, remaining):
        seen.append((seed, None if parent is None else parent["context"]))
        value = (0.8 if seed == "quick" else 0.4) if parent is None else (-1 if seed == "quick" else 1)
        return {"context": seed + " consequence", "tokens_used": 1, "value": value}

    shallow = search_futures(["quick", "careful"], rollout, lambda r, s: r["value"], max_depth=1)
    deep = search_futures(["quick", "careful"], rollout, lambda r, s: r["value"], max_depth=2)
    assert shallow.leaves[0]["seed"] == "quick"
    assert deep.leaves[0]["seed"] == "careful"
    assert deep.leaves[0]["parent_id"] is not None
    assert ("careful", "careful consequence") in seen
    assert all(n["epistemic_type"] == "hypothesis" for n in deep.nodes)


def test_search_limits_failures_tokens_and_deadline():
    failed = search_futures(list(range(50)), lambda *a: None, lambda *a: 0, max_nodes=3)
    assert failed.attempts == 3 and failed.budget_exhausted
    result = search_futures([1, 2], lambda *a: {"tokens_used": 1}, lambda *a: 1,
                            max_tokens=3, max_depth=5)
    assert result.tokens == 3 and result.budget_exhausted
    ticks = iter([0, 2])
    timed = search_futures([1], lambda *a: pytest.fail("ran past deadline"), lambda *a: 0,
                           budget_s=1, clock=lambda: next(ticks))
    assert timed.attempts == 0 and timed.budget_exhausted
    partial = search_futures([1, 2], lambda *a: {"failed": True, "tokens_used": 2},
                             lambda *a: pytest.fail("scored a failed rollout"), max_tokens=2)
    assert partial.tokens == 2 and partial.attempts == 1 and not partial.nodes


def test_imagination_has_no_feedback_writes_and_uses_small_vocab(tiny_model):
    hook = SteeringHook().attach(tiny_model, 1)
    hook.set_feedback(lambda h: pytest.fail("imagination wrote directly to bus"), every=1)
    stack = SubconsciousStack(16)
    spec = SpeculativeFutures(tiny_model, TinyTokenizer(), hook, stack, {},
                              rollout_tokens=3, rollout_budget_ms=10000, chained_continuation_tokens=0)
    before = {n: p.clone() for n, p in tiny_model.named_parameters()}
    result = spec._imagine(torch.zeros(16), torch.eye(16)[0], torch.tensor([[1, 2]]))
    assert result is not None and result["tokens_used"] == 3
    assert result["context_ids"].shape == (1, 5)
    assert result["predicted_state"].shape == (16,)
    assert torch.isfinite(result["vec"]).all()
    assert hook.feedback_enabled and not hook.active
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    hook.detach()


def test_evaluated_future_surfaces_then_expires(monkeypatch):
    from magnum_opus_v2.regions import subconscious as module
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    stack = SubconsciousStack(16, future_gain=10, l3_interpolate_top_2=False)
    stack.publish_futures([{"vec": torch.eye(16)[1], "id": 2, "depth": 2,
                            "name": "hypothetical consequence", "utility": 1}])
    stack.step(LatentBus(16), None, 0.05)
    assert stack.last_intrusive.source == "imagined_future"
    assert stack.last_intrusive.meta["epistemic_type"] == "hypothesis"
    now[0] += 4
    stack.step(LatentBus(16), None, 0.05)
    assert stack.last_intrusive.source != "imagined_future"
    assert stack.snapshot()["evaluated_futures"] == 0


def test_real_search_connects_to_fast_stack_and_skips_busy_model(tiny_model):
    hook = SteeringHook().attach(tiny_model, 1)
    stack = SubconsciousStack(16)
    spec = SpeculativeFutures(tiny_model, TinyTokenizer(), hook, stack, {},
                              rollout_tokens=2, chained_continuation_tokens=0,
                              rollout_budget_ms=10000, round_budget_ms=10000,
                              max_nodes=3, beam_width=1, branching_factor=2)
    spec.model_lock.acquire()
    assert spec.step(LatentBus(16), None, 1.5) is None
    spec.model_lock.release()
    assert spec.skipped_busy == 1
    # Even a due forecast cannot resolve from a reread of cached context.
    spec.ledger.horizon_s = 0
    spec.ledger.record([{"vec": torch.ones(16), "name": "prior", "probability": 0.9}], tick=0)
    spec.situation_provider = lambda: torch.ones(16)
    spec.step(LatentBus(16), None, 1.5)
    assert spec.snapshot()["search"]["depth_reached"] == 2
    assert stack.snapshot()["evaluated_futures"] > 0
    assert spec.ledger.metrics()["open"] > 0
    assert spec.ledger.metrics()["resolved"] == 0
    hook.detach()


def test_behavioral_evaluation_runs_matched_controls_and_keeps_weights(tiny_model):
    from magnum_opus_v2.evaluate import evaluate_directions
    before = {n: p.clone() for n, p in tiny_model.named_parameters()}
    report = evaluate_directions(tiny_model, TinyTokenizer(), make_profile(),
                                 [{"id": "x", "prompt": "ab", "target": "c", "contrast": "d"}],
                                 "calm", [0.5], [11, 23])
    row = report["rows"][0]
    assert len(row["random_margins"]) == 2
    assert row["positive_margin"] != row["negative_margin"]
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    assert not tiny_model.transformer.h[1]._forward_hooks


def test_replaced_profile_does_not_reuse_previous_dynamics(tmp_path):
    from magnum_opus_v2.profile import load_profile, save_profile
    profile = make_profile()
    profile.dynamics = {"fit": "old"}
    save_profile(profile, tmp_path)
    profile.dynamics = None
    save_profile(profile, tmp_path)
    restored = load_profile("tiny", tmp_path)
    assert restored.dynamics is None
    restored.validate_activation_site()


def test_legacy_checkpoint_rejected_before_mutating_runtime(tmp_path):
    from magnum_opus_v2.persistence import load_engine
    from types import SimpleNamespace
    path = tmp_path / "old.pt"
    torch.save({"engine_version": 1}, path)
    # Deliberately no region state: rejection must happen before any writes.
    engine = SimpleNamespace(profile=make_profile())
    with pytest.raises(RuntimeError, match="block-output"):
        load_engine(engine, path)


def test_new_perception_resolves_due_forecasts():
    from magnum_opus_v2.engine import V2Engine
    from magnum_opus_v2.forecast import ForecastLedger
    from types import SimpleNamespace
    ledger = ForecastLedger(horizon_s=0)
    truth = torch.eye(16)[0]
    ledger.record([{"vec": truth, "probability": 0.9}], tick=0)
    engine = object.__new__(V2Engine)
    engine.profile = make_profile()
    engine.speculative = SimpleNamespace(ledger=ledger)
    engine._embed_text = lambda text: truth
    engine.perceive_emotions("new report")
    assert ledger.metrics()["resolved"] == 1

"""Offline startup contracts: real block adapters, local setup, and safe profile reuse."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM,
    LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM,
    Qwen2Config, Qwen2ForCausalLM,
)

import magnum_opus_v2.profile as profiles
import magnum_opus_v2.startup as startup
from magnum_opus_v2.model_sources import (
    canonical_model_source, discover_cached_models, model_storage_key, validate_model_source,
)
from magnum_opus_v2.steering_hook import SteeringHook
from tests_engine.test_research_contracts import TinyTokenizer


def tiny_model():
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=128, n_embd=16,
                                     n_layer=2, n_head=2)).eval()


@pytest.mark.parametrize("family", ["gpt2", "llama", "qwen2", "gpt_neox", "opt"])
def test_real_model_adapters_accept_intervention_and_remove_hooks(family):
    common = dict(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                  num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)
    if family == "gpt2":
        model = tiny_model()
    elif family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**common, num_key_value_heads=2))
    elif family == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**common, num_key_value_heads=2))
    elif family == "gpt_neox":
        model = GPTNeoXForCausalLM(GPTNeoXConfig(**common))
    else:
        model = OPTForCausalLM(OPTConfig(**common, ffn_dim=32, word_embed_proj_dim=16))
    model.eval()
    startup.validate_model_boundary(model, TinyTokenizer(), "cpu")
    assert all(not block._forward_hooks for block in SteeringHook._layer_list(model))


def checkpoint(folder, *, shards=False):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}))
    if shards:
        (folder / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
        (folder / "part1.safetensors").write_bytes(b"fixture")
    else:
        (folder / "model.safetensors").write_bytes(b"fixture")
    return folder


def test_discovery_ignores_incomplete_shards_and_encoder_models(tmp_path):
    good = checkpoint(tmp_path / "models--org--chat" / "snapshots" / "abc123")
    broken = checkpoint(tmp_path / "models--org--partial" / "snapshots" / "def456", shards=True)
    encoder = checkpoint(tmp_path / "models--org--encoder" / "snapshots" / "ghi789")
    (encoder / "config.json").write_text('{"architectures": ["BertModel"]}')
    choices = discover_cached_models(tmp_path)
    assert len(choices) == 1 and choices[0]["source"] == canonical_model_source(str(good))
    (broken / "part2.safetensors").write_bytes(b"fixture")
    assert len(discover_cached_models(tmp_path)) == 2


@pytest.mark.parametrize("source", ["weights.gguf", "http://localhost:11434", "ollama:llama3"])
def test_unsupported_formats_explain_the_required_model_access(source):
    with pytest.raises(ValueError, match="activations"):
        validate_model_source(source)


def test_local_paths_are_canonical_and_artifact_names_are_portable(tmp_path, monkeypatch):
    first = checkpoint(tmp_path / "one" / "model")
    second = checkpoint(tmp_path / "two" / "model")
    monkeypatch.chdir(tmp_path)
    resolved = validate_model_source("one/model")
    assert resolved == canonical_model_source(str(first))
    assert model_storage_key(resolved) != model_storage_key(canonical_model_source(str(second)))
    assert model_storage_key("Qwen/Qwen2.5-3B-Instruct") == "Qwen--Qwen2.5-3B-Instruct"
    assert not any(c in model_storage_key(r"C:\Models\my model") for c in '/\\:<>"|?*')


def test_picker_accepts_cached_selection_and_requires_explicit_headless_model(monkeypatch):
    monkeypatch.setattr(startup, "discover_cached_models", lambda: [{"label": "fixture", "source": "picked"}])
    replies = iter(["9", "1"])
    assert startup.choose_model(interactive=True, input_fn=lambda _: next(replies)) == "picked"
    with pytest.raises(ValueError, match="non-interactive"):
        startup.choose_model(interactive=False)


def stub_calibration(monkeypatch):
    calls = []

    def create(source, profiles_dir, loaded_model, model_fingerprint, **kwargs):
        calls.append(loaded_model)
        profile = profiles.ModelProfile(
            profiles.ProfileMetadata(source, 1, 16, 2, ["calm"], "fixture",
                                     model_fingerprint=model_fingerprint),
            {"calm": torch.eye(16)[0]}, profiles.Baseline({"calm": 0}, 1),
        )
        profiles.save_profile(profile, profiles_dir)
        return profile

    monkeypatch.setattr(startup, "create_profile", create)
    return calls


def test_calibration_reuses_loaded_model_and_rebuilds_changed_identity(tmp_path, monkeypatch):
    calls = stub_calibration(monkeypatch)
    loaded = (tiny_model(), TinyTokenizer(), "cpu")
    first, rebuilt = startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    assert rebuilt and calls == [loaded]
    again, rebuilt = startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    assert not rebuilt and len(calls) == 1
    assert again.metadata.model_fingerprint == first.metadata.model_fingerprint
    loaded[0].config._commit_hash = "changed-revision"
    _, rebuilt = startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    assert rebuilt and len(calls) == 2
    archived = list((tmp_path / ".archive").glob("*/metadata.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["model_fingerprint"] == first.metadata.model_fingerprint


def test_failed_recalibration_preserves_existing_files(tmp_path, monkeypatch):
    stub_calibration(monkeypatch)
    loaded = (tiny_model(), TinyTokenizer(), "cpu")
    startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    before = {p.name: p.read_bytes() for p in (tmp_path / "fixture").iterdir()}

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted calibration")

    monkeypatch.setattr(startup, "create_profile", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path, rebuild=True)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "fixture").iterdir()}
    assert not (tmp_path / ".archive").exists()


def test_legacy_profile_is_archived_and_explicit_wrong_model_is_rejected(tmp_path, monkeypatch):
    stub_calibration(monkeypatch)
    loaded = (tiny_model(), TinyTokenizer(), "cpu")
    startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    meta = tmp_path / "fixture" / "metadata.json"
    data = json.loads(meta.read_text())
    data.pop("activation_site")
    data["version"] = "1.0"
    meta.write_text(json.dumps(data))
    _, rebuilt = startup.prepare_profile("fixture", loaded, profiles_dir=tmp_path)
    assert rebuilt
    with pytest.raises(ValueError, match="different model"):
        startup.prepare_profile("other", loaded, profile_path=tmp_path / "fixture")


def test_model_folder_is_not_mistaken_for_profile_folder(tmp_path):
    model_dir = checkpoint(tmp_path / "weights")
    source = canonical_model_source(str(model_dir))
    profile = profiles.ModelProfile(profiles.ProfileMetadata(source, 1, 16, 2, ["calm"], "fixture"),
                                    {"calm": torch.eye(16)[0]}, profiles.Baseline({"calm": 0}, 1))
    profiles.save_profile(profile, tmp_path / "profiles")
    assert profiles.load_profile(model_dir, tmp_path / "profiles").model_name == source


def test_actual_calibration_does_not_reload_or_train_llm(tmp_path, monkeypatch):
    model = tiny_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(profiles, "load_model", lambda *a, **kw: pytest.fail("Loaded a second model"))
    import magnum_opus_v2.mirror as mirror
    monkeypatch.setattr(mirror, "extract_dynamics", lambda *a, **kw: None)
    result = profiles.create_profile("tiny", profiles_dir=tmp_path, verbose=False,
                                     loaded_model=(model, TinyTokenizer(), "cpu"))
    assert len(result.vectors) == 12
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_loader_defaults_do_not_execute_custom_model_code_and_honors_offline(monkeypatch):
    import magnum_opus_v2.loader as loader
    calls = []
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    tok = SimpleNamespace(pad_token=None, eos_token="<eos>")
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *a, **kw: calls.append(kw) or tok)
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: calls.append(kw) or tiny_model())
    _, _, device = loader.load_model("fixture", "cpu", local_files_only=True)
    assert device == "cpu" and tok.pad_token == "<eos>"
    assert all(c["local_files_only"] and not c["trust_remote_code"] for c in calls)


def test_cuda_fallback_releases_failed_model_before_cpu_reload(monkeypatch):
    import weakref
    import magnum_opus_v2.loader as loader

    class TooLarge:
        is_quantized = False

        def to(self, device):
            raise torch.cuda.OutOfMemoryError("fixture OOM")

    references = []

    def load(*args, **kwargs):
        if not references:
            model = TooLarge()
            references.append(weakref.ref(model))
            return model
        assert references[0]() is None, "The failed GPU model is still retained"
        assert kwargs["torch_dtype"] == torch.float32
        return tiny_model()

    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(pad_token="eos"))
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loader.torch.cuda, "empty_cache", lambda: None)
    assert loader.load_model("fixture", local_files_only=True)[2] == "cpu"


def test_cli_model_listing_does_not_load_weights(monkeypatch, capsys):
    import compare_server
    monkeypatch.setattr(compare_server, "discover_cached_models", lambda: [])
    monkeypatch.setattr(compare_server, "load_model", lambda *a, **kw: pytest.fail("Loaded weights"))
    assert compare_server.main(["--list-models"]) == 0
    assert "No cached" in capsys.readouterr().out


def test_checkpoint_resume_requires_same_calibration_before_any_state_is_applied(tmp_path):
    from magnum_opus_v2 import V2Engine
    from magnum_opus_v2.persistence import load_engine, save_engine
    model = tiny_model()
    profile = profiles.ModelProfile(profiles.ProfileMetadata("tiny", 1, 16, 2, ["calm"], "fixture"),
                                    {"calm": torch.eye(16)[0]}, profiles.Baseline({"calm": 0}, 1))
    engine = V2Engine.from_profile(model, TinyTokenizer(), profile, device="cpu")
    path = tmp_path / "state.pt"
    try:
        engine.bus.tick_count = 123
        save_engine(engine, path)
        engine.bus.tick_count = 0
        assert load_engine(engine, path) and engine.bus.tick_count == 123
        profile.vectors["calm"] = torch.eye(16)[1]
        engine.bus.tick_count = 0
        with pytest.raises(RuntimeError, match="calibration"):
            load_engine(engine, path)
        assert engine.bus.tick_count == 0
    finally:
        engine.stop()

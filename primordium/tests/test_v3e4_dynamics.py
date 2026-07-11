"""V3 Era 4, M3 — affect dynamics measured from its own life. The
audit found onset/decay were authored constants sold as 'measured';
now they are fitted from each affect's lived activation history, and
the signature says honestly whether it is still running on defaults."""

import numpy as np
import torch

from primordium.config import OrganismConfig
from primordium.mind.valence import ValenceCompass, fit_dynamics


def _ar1(rho, n, jump=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + jump * rng.standard_normal()
    return x


def test_fit_recovers_known_dynamics():
    slow = fit_dynamics(_ar1(0.95, 600, seed=1), dt=0.2)
    fast = fit_dynamics(_ar1(0.30, 600, seed=2), dt=0.2)
    assert slow is not None and fast is not None
    o_s, d_s, m_s = slow
    o_f, d_f, m_f = fast
    assert abs(m_s["rho1"] - 0.95) < 0.05        # the fit sees the truth
    assert abs(m_f["rho1"] - 0.30) < 0.12
    assert d_f > d_s * 2                         # fast feelings fade faster
    assert 0.01 <= d_s <= 0.5 and 0.05 <= o_f <= 0.8

    assert fit_dynamics([0.0] * 500, dt=0.2) is None   # flat says nothing
    assert fit_dynamics([1, 2], dt=0.2) is None        # too short to claim


def test_compass_collects_and_fits_lived_activation():
    c = ValenceCompass(dim=384)
    assert c.fitted_dynamics("affect_0", dt=0.2) is None
    for v in _ar1(0.9, 300, seed=3):
        c.observe_activation("affect_0", float(v))
    fd = c.fitted_dynamics("affect_0", dt=0.2)
    assert fd is not None
    _onset, _decay, meta = fd
    assert meta["n"] == 300 and abs(meta["rho1"] - 0.9) < 0.08


def test_installed_affects_get_their_own_measured_dynamics():
    """Two affects with different lived time-constants must end up with
    DIFFERENT EmotionConfigs — and the signature must say 'fitted'.
    A third with no history stays on disclosed defaults."""
    from primordium.mind.coupling import CompassRegion
    from primordium.mind.plastic import MoodField, Undertow
    from primordium.mind.tide import Tide
    from magnum_opus_v2.bus import LatentBus
    from magnum_opus_v2.config import BusConfig
    from magnum_opus_v2.regions import NoiseSampler

    cfg = OrganismConfig()
    d = cfg.d_model
    compass = ValenceCompass(dim=d)
    for i, name in enumerate(("affect_slow", "affect_fast", "affect_new")):
        compass.installed[name] = torch.randn(d)
        compass.provenance[name] = "lived"
    for v in _ar1(0.95, 400, seed=4):
        compass.observe_activation("affect_slow", float(v))
    for v in _ar1(0.30, 400, seed=5):
        compass.observe_activation("affect_fast", float(v))

    mood = MoodField(emotion_vectors={}, device="cpu",
                     steering_strength=1.0)
    subc = Undertow(hidden_dim=d, device="cpu",
                    samplers=[NoiseSampler(d, "cpu", magnitude=1.0)],
                    l3_perturbation_strength=0.4, emotion_vectors=None)
    bus = LatentBus(hidden_dim=d, device="cpu", config=BusConfig())

    class _W:
        stage = 0
        tick_id = 0

    class _Board(dict):
        def put(self, **kv): self.update(kv)
        def get(self, k, dflt=None): return dict.get(self, k, dflt)

    board = _Board()
    board.put(affects_dirty=True)                # force a refresh
    region = CompassRegion(compass, mood, subc, bus, board,
                           cfg=cfg, worker=_W())
    region.step(bus, Tide(cfg), 1.0)

    dyn = {n: compass.signatures[n]["dynamics"]
           for n in ("affect_slow", "affect_fast", "affect_new")}
    assert dyn["affect_slow"]["mode"] == "fitted"
    assert dyn["affect_fast"]["mode"] == "fitted"
    assert dyn["affect_new"]["mode"] == "default"     # honesty until data
    assert dyn["affect_fast"]["decay"] > dyn["affect_slow"]["decay"] * 2
    # and the MoodField actually received them
    st = mood._state.configs
    assert abs(st["affect_fast"].decay_rate
               - dyn["affect_fast"]["decay"]) < 1e-3
    assert abs(st["affect_slow"].decay_rate
               - dyn["affect_slow"]["decay"]) < 1e-3


def test_stability_bar_matches_the_docstring():
    """Install only after surviving re-estimation TWICE (streak >= 2)."""
    c = ValenceCompass(dim=8, min_samples=1)
    vec = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
    cand = {"aff": {"vec": vec, "sig": {"drives": {}, "valence": 0.0}}}
    assert c.stabilize_and_install(dict(cand)) == []      # first sighting
    assert c.stabilize_and_install(dict(cand)) == []      # streak 1
    assert c.stabilize_and_install(dict(cand)) == ["aff"]  # streak 2: real

"""Hygiene: the engine's code speaks in mechanism, not in the private
design conversation that produced it. No developer-conversation quotes,
no personal attributions, no internal process labels leaking into the
shipped source."""

import re
from pathlib import Path

# Phrases that would mean a private conversation or an internal process
# label leaked into shipped code. Kept to exactly what this pass removed,
# so the guard stays honest and falsifiable.
BANNED = [
    "the user described",
    "as i described",
    "as we described",
    "design conversation",
    "blueprint",
    "era-4 audit",
    "era 6",
    "adr-002",
    "adr-001",
]

ENGINE = Path("magnum_opus_v2")


def _sources():
    return list(ENGINE.rglob("*.py"))


def test_no_personal_attribution_or_process_labels():
    offenders = []
    for p in _sources():
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in BANNED:
            if phrase in text:
                offenders.append((str(p), phrase))
    assert not offenders, f"banned phrases in engine source: {offenders}"


def test_bus_playground_is_impersonal():
    text = (ENGINE / "bus.py").read_text(encoding="utf-8")
    assert "the user described" not in text
    # the mechanism sentence survives the rewrite
    assert re.search(r"regions are never sequenced", text)

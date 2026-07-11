"""
Webtext — a dormant organ. THERE IS NO IMPLEMENTATION BEHIND THIS GATE.

This module is deliberately an interface and nothing else: no sockets,
no HTTP client, no parsing, no imports that could reach a network. If
the webtext gate is ever unlocked (stage 2 + a long, stable caregiver
era + the creator's word), the implementation gets written THEN, in the
open, with the allowlist and budgets below as its contract — not
switched on from a hidden code path.

The organism can feel this limb exists (its [present, open] flags are
in interoception) and will find only GateLockedError at the end of it.
"""

from primordium.safety.gatehouse import GateLockedError


class WebtextPort:
    """Interface contract for a future, gated text-from-the-web sense.

    ALLOWLIST         only these domains would ever be reachable
    MAX_BYTES_PER_DAY hard daily budget, enforced outside the organism
    TEXT_ONLY         no executable content, no scripts, no media
    """

    ALLOWLIST: tuple = ()          # empty until an unlock ceremony fills it
    MAX_BYTES_PER_DAY: int = 0
    TEXT_ONLY: bool = True

    def __init__(self, gatehouse):
        self._gatehouse = gatehouse

    def fetch(self, url: str) -> str:
        raise GateLockedError(
            "webtext is a dormant organ: no implementation exists behind "
            "this gate. Milestones + the creator's word would come first; "
            "then the implementation is written in the open.")

"""Drive a MolmoSpaces episode from an external process.

``BridgePolicy`` is a ``BasePolicy`` that forwards a typed, role-tagged episode
protocol to any process that speaks msgpack over a websocket; ``client.py`` is
the reference implementation of the other side, and is meant to be COPIED into
the external stack rather than imported from here.
"""

from molmo_spaces.bridge.policy import BridgePolicy
from molmo_spaces.bridge.protocol import PROTOCOL, ROLES, BridgeError

__all__ = ["PROTOCOL", "ROLES", "BridgeError", "BridgePolicy"]

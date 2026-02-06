"""Message bus module for decoupled channel-agent communication."""

from krolik.bus.events import InboundMessage, OutboundMessage
from krolik.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]

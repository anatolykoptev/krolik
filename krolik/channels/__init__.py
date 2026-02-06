"""Chat channels module with plugin architecture."""

from krolik.channels.base import BaseChannel
from krolik.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]

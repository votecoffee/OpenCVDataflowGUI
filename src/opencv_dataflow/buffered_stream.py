"""Lifecycle interface for blocks that retain input and emit delayed output."""

from abc import ABC, abstractmethod

from .core import Block


class BufferedStreamBlock(Block, ABC):
    """The scheduler sets end_of_input on the last call of a finished image stream.

    pending_output keeps finite playback alive while delayed output remains.
    Implement process({}) to drain when end_of_input is true. Runtime snapshots
    must preserve buffered packets, including their original tickets and times.
    """

    buffered_stream = True
    end_of_input = False
    stream_inputs = ("image",)
    poll_without_input = True
    requires_serial_execution = True
    outcome_passthrough = True
    per_packet_outcomes = True
    report_emitted_tickets_only = True

    @property
    @abstractmethod
    def pending_output(self):
        """Whether accepted input still has output waiting to be emitted."""

    @property
    @abstractmethod
    def pending_tickets(self):
        """Tickets awaiting output; return an empty tuple for unticketed streams."""

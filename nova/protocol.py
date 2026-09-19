"""Control-plane message types. JSON over Pear or TCP. Not the PNG data plane."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

HELLO = "HELLO"
NODE_MANIFEST = "NODE_MANIFEST"
NODE_UPDATE = "NODE_UPDATE"
HEARTBEAT = "HEARTBEAT"
NODE_DISCONNECTED = "NODE_DISCONNECTED"
NODE_GOODBYE = "NODE_GOODBYE"
NODE_OFFLINE = "NODE_OFFLINE"
NODE_RECOVERED = "NODE_RECOVERED"

JOB_ANNOUNCE = "JOB_ANNOUNCE"

WORK_REQUEST = "WORK_REQUEST"
TASK_OFFER = "TASK_OFFER"
TASK_ACCEPT = "TASK_ACCEPT"
TASK_REJECT = "TASK_REJECT"

TASK_STARTED = "TASK_STARTED"
TASK_PROGRESS = "TASK_PROGRESS"
TASK_COMPLETE = "TASK_COMPLETE"
TASK_FAILED = "TASK_FAILED"

RESULT_ACK = "RESULT_ACK"


class Envelope(BaseModel):
    type: str
    from_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_version: int = PROTOCOL_VERSION

    def encode(self) -> bytes:
        return (self.model_dump_json() + "\n").encode("utf-8")

    @classmethod
    def decode(cls, line: str | bytes) -> Envelope:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        return cls.model_validate_json(line.strip())


def msg(type_: str, from_id: str, **payload: Any) -> Envelope:
    return Envelope(type=type_, from_id=from_id, payload=payload)

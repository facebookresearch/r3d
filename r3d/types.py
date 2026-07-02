# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    AI = "assistant"
    SYSTEM = "system"


class MessageAttachmentType(str, Enum):
    BASE64_IMAGE = "base64_image"


@dataclass
class MessageAttachment:
    type: MessageAttachmentType
    data: str
    mime: str = "image/jpeg"


@dataclass
class Message:
    text: str
    role: ChatRole
    attachments: list[MessageAttachment] = field(default_factory=list)

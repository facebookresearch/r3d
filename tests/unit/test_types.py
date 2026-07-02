# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

from r3d.types import ChatRole, Message, MessageAttachment, MessageAttachmentType


class TestChatRole(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(ChatRole.USER.value, "user")
        self.assertEqual(ChatRole.ASSISTANT.value, "assistant")
        self.assertEqual(ChatRole.AI.value, "assistant")
        self.assertEqual(ChatRole.SYSTEM.value, "system")

    def test_ai_equals_assistant(self) -> None:
        self.assertEqual(ChatRole.AI, ChatRole.ASSISTANT)

    def test_str_enum(self) -> None:
        self.assertIsInstance(ChatRole.USER, str)
        self.assertEqual(str(ChatRole.USER), "ChatRole.USER")


class TestMessageAttachmentType(unittest.TestCase):
    def test_base64_image(self) -> None:
        self.assertEqual(MessageAttachmentType.BASE64_IMAGE.value, "base64_image")


class TestMessageAttachment(unittest.TestCase):
    def test_construction(self) -> None:
        att = MessageAttachment(
            type=MessageAttachmentType.BASE64_IMAGE,
            data="abc123",
        )
        self.assertEqual(att.type, MessageAttachmentType.BASE64_IMAGE)
        self.assertEqual(att.data, "abc123")
        self.assertEqual(att.mime, "image/jpeg")

    def test_custom_mime(self) -> None:
        att = MessageAttachment(
            type=MessageAttachmentType.BASE64_IMAGE,
            data="xyz",
            mime="image/png",
        )
        self.assertEqual(att.mime, "image/png")


class TestMessage(unittest.TestCase):
    def test_construction_minimal(self) -> None:
        msg = Message(text="hello", role=ChatRole.USER)
        self.assertEqual(msg.text, "hello")
        self.assertEqual(msg.role, ChatRole.USER)
        self.assertEqual(msg.attachments, [])

    def test_construction_with_attachments(self) -> None:
        att = MessageAttachment(
            type=MessageAttachmentType.BASE64_IMAGE,
            data="img_data",
        )
        msg = Message(text="look", role=ChatRole.ASSISTANT, attachments=[att])
        self.assertEqual(len(msg.attachments), 1)
        self.assertEqual(msg.attachments[0].data, "img_data")

    def test_default_attachments_independent(self) -> None:
        msg1 = Message(text="a", role=ChatRole.USER)
        msg2 = Message(text="b", role=ChatRole.USER)
        msg1.attachments.append(
            MessageAttachment(type=MessageAttachmentType.BASE64_IMAGE, data="x")
        )
        self.assertEqual(len(msg2.attachments), 0)


if __name__ == "__main__":
    unittest.main()

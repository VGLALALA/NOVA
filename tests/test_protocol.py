from nova.protocol import PROTOCOL_VERSION, Envelope, msg


def test_protocol_version_is_one() -> None:
    assert PROTOCOL_VERSION == 1


def test_msg_sets_type_and_payload() -> None:
    env = msg("HELLO", "nova-abc", node_id="nova-abc")
    assert env.type == "HELLO"
    assert env.from_id == "nova-abc"
    assert env.payload == {"node_id": "nova-abc"}
    assert env.protocol_version == PROTOCOL_VERSION


def test_envelope_encode_decode_roundtrip() -> None:
    original = msg("TASK_COMPLETE", "nova-w1", sha256="deadbeef", lease_gen=2)
    decoded = Envelope.decode(original.encode())
    assert decoded.type == original.type
    assert decoded.from_id == original.from_id
    assert decoded.payload == original.payload
    assert decoded.protocol_version == original.protocol_version
    assert decoded.ts == original.ts

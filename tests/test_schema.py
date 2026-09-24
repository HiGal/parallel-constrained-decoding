import pytest

from pcd import Field, Schema, SchemaError


def test_presets_parse(presets):
    for name, p in presets.items():
        s = Schema.coerce(p)  # whole preset file: {"schema": ..., "context": ...}
        assert len(s) == len(p["schema"])
        assert s.names == list(p["schema"])
        assert s.waves == [s.names]


def test_boolean_and_enum_labels():
    s = Schema.from_dict({
        "flag": {"type": "boolean", "description": "  a\n  flag  "},
        "tier": {"type": "choice", "choices": ["A", "B"]},
        "stars": {"type": "enum", "choices": [1, 2, 3]},
    })
    assert s["flag"].labels == (True, False)
    assert s["flag"].description == "a flag"
    assert s["tier"].type == "enum" and s["tier"].labels == ("A", "B")
    assert s["stars"].labels == (1, 2, 3)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"x": {"type": "enum", "choices": []}}, "at least one choice"),
        ({"x": {"type": "enum", "choices": ["A", "A"]}}, "duplicate choices"),
        ({"x": {"type": "enum", "choices": ["A", True]}}, "strings or numbers"),
        ({"x": {"type": "number"}}, "unsupported type"),
        ({"x": {"type": "boolean", "choices": ["yes"]}}, "must not define choices"),
        ({"x": {"type": "boolean", "depends_on": ["y"]}}, "unknown field"),
        ({"x": {"type": "boolean", "depends_on": ["x"]}}, "depends on itself"),
        ({}, "at least one field"),
    ],
)
def test_invalid_schemas(spec, message):
    with pytest.raises(SchemaError, match=message):
        Schema.from_dict(spec)


def test_dependency_waves_and_cycles():
    s = Schema.from_dict({
        "refund_tier": {"type": "enum", "choices": ["NONE", "FULL"], "depends_on": ["refund"]},
        "refund": {"type": "boolean"},
        "sentiment": {"type": "enum", "choices": ["POS", "NEG"]},
        "note": {"type": "boolean", "depends_on": ["refund_tier"]},
    })
    assert s.waves == [["refund", "sentiment"], ["refund_tier"], ["note"]]
    with pytest.raises(SchemaError, match="cycle"):
        Schema.from_dict({
            "a": {"type": "boolean", "depends_on": ["b"]},
            "b": {"type": "boolean", "depends_on": ["a"]},
        })


def test_fingerprint_is_stable_and_order_sensitive():
    a = Schema.from_dict({"x": {"type": "boolean"}, "y": {"type": "enum", "choices": ["A", "B"]}})
    b = Schema.from_dict({"x": {"type": "boolean"}, "y": {"type": "enum", "choices": ["A", "B"]}})
    c = Schema.from_dict({"y": {"type": "enum", "choices": ["A", "B"]}, "x": {"type": "boolean"}})
    assert a.fingerprint == b.fingerprint != c.fingerprint


def test_json_schema_and_pydantic():
    js = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "description": "Is it ok"},
            "level": {"enum": ["LOW", "HIGH"]},
            "mode": {"anyOf": [{"const": "fast"}, {"const": "slow"}]},
            "color": {"$ref": "#/$defs/Color"},
        },
        "$defs": {"Color": {"enum": ["red", "green"], "type": "string"}},
    }
    s = Schema.coerce(js)
    assert [f.type for f in s] == ["boolean", "enum", "enum", "enum"]
    assert s["mode"].choices == ("fast", "slow") and s["color"].choices == ("red", "green")
    with pytest.raises(SchemaError, match="neither a boolean nor an enum"):
        Schema.from_json_schema({"properties": {"n": {"type": "integer"}}})

    pydantic = pytest.importorskip("pydantic")
    from enum import Enum
    from typing import Literal

    class Priority(str, Enum):
        P0 = "P0"
        P1 = "P1"

    class Ticket(pydantic.BaseModel):
        urgent: bool = pydantic.Field(description="Needs attention now")
        priority: Priority
        team: Literal["billing", "infra"]

    t = Schema.coerce(Ticket)
    assert t.names == ["urgent", "priority", "team"]
    assert t["urgent"].description == "Needs attention now"
    assert t["priority"].choices == ("P0", "P1") and t["team"].choices == ("billing", "infra")


def test_validate_values():
    s = Schema([Field("ok", "boolean"), Field("lvl", "enum", choices=("LOW", "HIGH")), Field("n", "enum", choices=(1, 2))])
    assert s.validate_values({"ok": True, "lvl": "LOW", "n": 2}) == ([], [], [])
    missing, extra, invalid = s.validate_values({"ok": "true", "lvl": "MID", "zzz": 1})
    assert missing == ["n"] and extra == ["zzz"] and invalid == ["ok='true'", "lvl='MID'"]

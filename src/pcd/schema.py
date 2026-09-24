"""Schemas: the closed set of decisions the engine makes.

A schema is an ordered list of fields. Every field has a finite set of labels:

* ``boolean`` fields have the labels ``True`` and ``False``;
* ``enum`` fields have an explicit list of choices (strings or numbers).

Free text, numbers without a closed set and nested objects are out of scope:
parallel constrained decoding only works when every value can be enumerated.

A field may declare ``depends_on``. Dependent fields are decided in a later
*wave*, with the answers of earlier waves written into the prompt, so that
related fields can stay consistent (notebook 05, section 5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence, Union

BOOLEAN = "boolean"
ENUM = "enum"

_TYPE_ALIASES = {
    "boolean": BOOLEAN,
    "bool": BOOLEAN,
    "enum": ENUM,
    "choice": ENUM,
    "selection": ENUM,
    "categorical": ENUM,
}

Label = Union[bool, str, int, float]


class SchemaError(ValueError):
    """Raised for schemas the engine cannot decode."""


@dataclass(frozen=True)
class Field:
    """One decision: a key in the output JSON and its allowed labels."""

    name: str
    type: str = ENUM
    description: str = ""
    choices: tuple = ()
    depends_on: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError(f"field name must be a non-empty string, got {self.name!r}")
        kind = _TYPE_ALIASES.get(str(self.type).lower())
        if kind is None:
            raise SchemaError(
                f"field {self.name!r}: unsupported type {self.type!r}; use 'boolean' or 'enum'"
            )
        object.__setattr__(self, "type", kind)
        object.__setattr__(self, "description", " ".join(str(self.description or "").split()))
        object.__setattr__(self, "depends_on", tuple(self.depends_on or ()))

        if kind == BOOLEAN:
            if self.choices:
                raise SchemaError(f"boolean field {self.name!r} must not define choices")
            return

        choices = tuple(self.choices or ())
        if not choices:
            raise SchemaError(f"enum field {self.name!r} needs at least one choice")
        for c in choices:
            if isinstance(c, bool) or not isinstance(c, (str, int, float)):
                raise SchemaError(
                    f"field {self.name!r}: choices must be strings or numbers, got {c!r}"
                )
            if isinstance(c, str) and not c.strip():
                raise SchemaError(f"field {self.name!r}: empty choice")
        rendered = [render_label(c) for c in choices]
        if len(set(rendered)) != len(rendered):
            dupes = sorted({r for r in rendered if rendered.count(r) > 1})
            raise SchemaError(f"field {self.name!r}: duplicate choices {dupes}")
        object.__setattr__(self, "choices", choices)

    @property
    def labels(self) -> tuple:
        """The values this field can take, in order."""
        return (True, False) if self.type == BOOLEAN else self.choices

    @property
    def cardinality(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.type == ENUM:
            d["choices"] = list(self.choices)
        if self.depends_on:
            d["depends_on"] = list(self.depends_on)
        return d


def render_label(label: Label) -> str:
    """How a label is written in the JSON output (``true``, ``"HIGH"``, ``3``)."""
    return json.dumps(label, ensure_ascii=False)


class Schema:
    """An ordered collection of fields."""

    def __init__(self, fields: Iterable[Field], name: str | None = None):
        self.name = name
        self.fields: dict[str, Field] = {}
        for f in fields:
            if f.name in self.fields:
                raise SchemaError(f"duplicate field {f.name!r}")
            self.fields[f.name] = f
        if not self.fields:
            raise SchemaError("a schema needs at least one field")
        for f in self.fields.values():
            for dep in f.depends_on:
                if dep not in self.fields:
                    raise SchemaError(f"field {f.name!r} depends on unknown field {dep!r}")
                if dep == f.name:
                    raise SchemaError(f"field {f.name!r} depends on itself")
        self._waves = self._compute_waves()
        self._fingerprint: str | None = None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_dict(cls, spec: Mapping[str, Mapping[str, Any]], name: str | None = None) -> "Schema":
        """Build from ``{"field": {"type": ..., "description": ..., "choices": [...]}}``.

        This is the preset format of the original project (``presets/*.json``).
        """
        fields = []
        for fname, fspec in spec.items():
            if not isinstance(fspec, Mapping):
                raise SchemaError(f"field {fname!r}: expected a mapping, got {type(fspec).__name__}")
            fields.append(
                Field(
                    name=fname,
                    type=fspec.get("type", ENUM),
                    description=fspec.get("description", ""),
                    choices=tuple(fspec.get("choices") or ()),
                    depends_on=tuple(fspec.get("depends_on") or ()),
                )
            )
        return cls(fields, name=name)

    @classmethod
    def from_json_schema(cls, js: Mapping[str, Any], name: str | None = None) -> "Schema":
        """Build from a JSON Schema object whose properties are booleans or enums.

        Supports ``{"type": "boolean"}``, ``{"enum": [...]}``, ``{"const": x}``,
        ``anyOf``/``oneOf`` of consts, and ``$ref`` into ``$defs``/``definitions``
        (which is what Pydantic emits for ``Literal[...]`` and ``Enum`` types).
        """
        defs = {**js.get("definitions", {}), **js.get("$defs", {})}

        def resolve(node: Mapping[str, Any]) -> Mapping[str, Any]:
            ref = node.get("$ref")
            if ref:
                key = ref.rsplit("/", 1)[-1]
                if key not in defs:
                    raise SchemaError(f"unresolvable $ref {ref!r}")
                merged = {**resolve(defs[key]), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            if "allOf" in node and len(node["allOf"]) == 1:
                return {**resolve(node["allOf"][0]), **{k: v for k, v in node.items() if k != "allOf"}}
            return node

        props = js.get("properties")
        if not isinstance(props, Mapping) or not props:
            raise SchemaError("JSON schema must be an object with 'properties'")
        fields = []
        for fname, raw in props.items():
            node = resolve(raw)
            desc = node.get("description") or node.get("title") or ""
            deps = tuple(node.get("x-depends-on") or node.get("depends_on") or ())
            if node.get("type") == "boolean":
                fields.append(Field(fname, BOOLEAN, desc, depends_on=deps))
                continue
            choices = None
            if "enum" in node:
                choices = list(node["enum"])
            elif "const" in node:
                choices = [node["const"]]
            else:
                alts = node.get("anyOf") or node.get("oneOf")
                if alts:
                    alts = [resolve(a) for a in alts]
                    if all("const" in a for a in alts):
                        choices = [a["const"] for a in alts]
                    elif all("enum" in a for a in alts):
                        choices = [c for a in alts for c in a["enum"]]
            if not choices:
                raise SchemaError(
                    f"property {fname!r} is neither a boolean nor an enum; "
                    "parallel constrained decoding needs a closed set of values"
                )
            if all(isinstance(c, bool) for c in choices) and set(choices) == {True, False}:
                fields.append(Field(fname, BOOLEAN, desc, depends_on=deps))
            else:
                fields.append(Field(fname, ENUM, desc, tuple(choices), depends_on=deps))
        return cls(fields, name=name or js.get("title"))

    @classmethod
    def from_pydantic(cls, model: Any) -> "Schema":
        """Build from a Pydantic v2 model whose fields are ``bool``, ``Literal`` or ``Enum``."""
        if not hasattr(model, "model_json_schema"):
            raise SchemaError("expected a Pydantic v2 model class")
        return cls.from_json_schema(model.model_json_schema(), name=model.__name__)

    @classmethod
    def coerce(cls, obj: Any) -> "Schema":
        """Accept a Schema, a preset-style dict, a JSON Schema dict, or a Pydantic model."""
        if isinstance(obj, Schema):
            return obj
        if isinstance(obj, Mapping):
            if "properties" in obj and isinstance(obj.get("properties"), Mapping):
                return cls.from_json_schema(obj)
            if "schema" in obj and isinstance(obj["schema"], Mapping):  # a whole preset file
                return cls.from_dict(obj["schema"], name=obj.get("id"))
            return cls.from_dict(obj)
        if hasattr(obj, "model_json_schema"):
            return cls.from_pydantic(obj)
        if isinstance(obj, Sequence) and obj and all(isinstance(f, Field) for f in obj):
            return cls(obj)
        raise SchemaError(f"cannot build a schema from {type(obj).__name__}")

    # ------------------------------------------------------------------ accessors

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields.values())

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    @property
    def names(self) -> list[str]:
        return list(self.fields)

    @property
    def waves(self) -> list[list[str]]:
        """Fields grouped into decoding waves; each wave only depends on earlier ones."""
        return [list(w) for w in self._waves]

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            blob = json.dumps(self.to_dict(), sort_keys=False, ensure_ascii=False)
            self._fingerprint = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self._fingerprint

    def to_dict(self) -> dict:
        return {name: f.to_dict() for name, f in self.fields.items()}

    def validate_values(self, values: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
        """Check a decoded object: returns (missing keys, extra keys, invalid values)."""
        missing = [n for n in self.fields if n not in values]
        extra = [k for k in values if k not in self.fields]
        invalid = []
        for n, f in self.fields.items():
            if n not in values:
                continue
            v = values[n]
            ok = isinstance(v, bool) if f.type == BOOLEAN else (
                not isinstance(v, bool) and any(v == c and type(v) is type(c) for c in f.choices)
            )
            if not ok:
                invalid.append(f"{n}={v!r}")
        return missing, extra, invalid

    # ------------------------------------------------------------------ internals

    def _compute_waves(self) -> list[tuple[str, ...]]:
        level: dict[str, int] = {}
        visiting: set[str] = set()

        def depth(n: str) -> int:
            if n in level:
                return level[n]
            if n in visiting:
                raise SchemaError(f"dependency cycle through field {n!r}")
            visiting.add(n)
            deps = self.fields[n].depends_on
            level[n] = 0 if not deps else 1 + max(depth(d) for d in deps)
            visiting.discard(n)
            return level[n]

        for n in self.fields:
            depth(n)
        n_waves = 1 + max(level.values())
        return [tuple(n for n in self.fields if level[n] == w) for w in range(n_waves)]

    def __repr__(self) -> str:
        return f"Schema({len(self)} fields, {len(self._waves)} wave(s))"

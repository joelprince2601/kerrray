"""Low-level parsing primitives behind :mod:`kerrray.utils.config`.

These helpers turn raw YAML values into the typed fields of the configuration
dataclasses and produce the error messages users see. They know nothing about
the schema itself; the schema lives in :mod:`kerrray.utils.config_blocks` and
:mod:`kerrray.utils.config`. Experiment drivers reuse :func:`parse_block` for
their own frozen parameter dataclasses (docs/decisions.md, D-009), so the
converter table covers scalars, booleans, optional floats and homogeneous
lists.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Final, TypeVar

import yaml

_T = TypeVar("_T")

Converter = Callable[[Any, str], Any]
"""``converter(raw_value, key)`` returns the typed value or raises :class:`ConfigError`."""


class ConfigError(ValueError):
    """Raised when a configuration is malformed or violates a constraint."""


def as_float(value: Any, key: str) -> float:
    """Coerce an int, float or numeric string to ``float``; booleans are rejected."""
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a number, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"'{key}' must be a number, got {value!r}") from None
    raise ConfigError(f"'{key}' must be a number, got {type(value).__name__}")


def as_int(value: Any, key: str) -> int:
    """Coerce an int, integral float or integer string to ``int``; booleans are rejected."""
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer, got boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"'{key}' must be an integer, got {value!r}") from None
    raise ConfigError(f"'{key}' must be an integer, got {type(value).__name__}")


def as_str(value: Any, key: str) -> str:
    """Require a string value."""
    if not isinstance(value, str):
        raise ConfigError(f"'{key}' must be a string, got {type(value).__name__}")
    return value


def as_bool(value: Any, key: str) -> bool:
    """Require a YAML boolean (``true`` / ``false``); strings and numbers are rejected."""
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean (true/false), got {value!r}")
    return value


def as_optional_float(value: Any, key: str) -> float | None:
    """``null`` (``None``) stays ``None``; anything else must satisfy :func:`as_float`."""
    if value is None:
        return None
    return as_float(value, key)


def _list_of(item: Converter, label: str) -> Converter:
    """Build a converter for a YAML sequence whose items pass through ``item``."""

    def convert(value: Any, key: str) -> list[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{key}' must be a list of {label}, got {type(value).__name__}")
        return [item(element, f"{key}[{index}]") for index, element in enumerate(value)]

    return convert


def _tuple_of(item: Converter, label: str) -> Converter:
    as_list = _list_of(item, label)

    def convert(value: Any, key: str) -> tuple[Any, ...]:
        return tuple(as_list(value, key))

    return convert


as_float_list: Final[Converter] = _list_of(as_float, "numbers")
as_int_list: Final[Converter] = _list_of(as_int, "integers")
as_str_list: Final[Converter] = _list_of(as_str, "strings")


def to_plain(value: Any) -> Any:
    """Return a deep copy of nested data built only from ``dict``, ``list`` and scalars.

    Every :class:`~collections.abc.Mapping` (including a read-only
    ``MappingProxyType``) becomes a ``dict`` and every ``list`` or ``tuple``
    becomes a ``list``, so the result is safe to mutate and can be written as
    YAML or JSON. Scalars are returned unchanged.
    """
    if isinstance(value, Mapping):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def as_mapping(value: Any, key: str) -> dict[str, Any]:
    """Require a mapping value and return a plain, deep-copied ``dict`` (see :func:`to_plain`)."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return to_plain(value)


CONVERTERS: Final[dict[str, Converter]] = {
    "float": as_float,
    "int": as_int,
    "str": as_str,
    "bool": as_bool,
    "float | None": as_optional_float,
    "Optional[float]": as_optional_float,
    "list[float]": as_float_list,
    "list[int]": as_int_list,
    "list[str]": as_str_list,
    "tuple[float, ...]": _tuple_of(as_float, "numbers"),
    "tuple[int, ...]": _tuple_of(as_int, "integers"),
    "tuple[str, ...]": _tuple_of(as_str, "strings"),
    "dict[str, Any]": as_mapping,
    "Mapping[str, Any]": as_mapping,
}
"""Converter for each dataclass field annotation the schema and the experiment
parameter blocks use.

Annotations are matched by their text with whitespace and a ``typing.`` prefix
removed, so ``float|None``, ``float | None`` and ``typing.Optional[float]``
all select the optional-float converter, whether the annotation is a string
(``from __future__ import annotations``) or a runtime type object. A new
field type needs an entry here before :func:`parse_block` can read it.
"""


def _canonical(name: str) -> str:
    return "".join(name.split()).replace("typing.", "")


_CANONICAL_CONVERTERS: Final[dict[str, Converter]] = {
    _canonical(name): converter for name, converter in CONVERTERS.items()
}


def _annotation_name(spec: Field[Any]) -> str:
    annotation = spec.type
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return annotation.__name__
    return repr(annotation)


def converter_for(spec: Field[Any], key: str) -> Converter:
    """Return the :data:`CONVERTERS` entry for dataclass field ``spec``.

    Raises:
        TypeError: If the annotation has no converter. This is a schema
            programming error, not a user configuration error, hence not a
            :class:`ConfigError`.
    """
    type_name = _annotation_name(spec)
    converter = _CANONICAL_CONVERTERS.get(_canonical(type_name))
    if converter is None:
        raise TypeError(
            f"no converter for field {key!r} annotated {type_name!r}; "
            "extend CONVERTERS in kerrray.utils.config_parsing"
        )
    return converter


def validate_block_type(block_type: type) -> None:
    """Check once that ``block_type`` is a dataclass whose every field has a converter.

    :mod:`kerrray.utils.config` calls this at import time for each block so a
    misannotated field fails when the package is imported, not when a user
    happens to set that key.
    """
    if not is_dataclass(block_type):
        raise TypeError(f"{block_type!r} is not a dataclass")
    for spec in fields(block_type):
        converter_for(spec, f"{block_type.__name__}.{spec.name}")


def reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    """Raise :class:`ConfigError` naming every key of ``data`` not in ``allowed``."""
    allowed_keys = list(allowed)
    unknown = sorted(str(key) for key in data if key not in allowed_keys)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {path}: {', '.join(unknown)} (allowed: {', '.join(allowed_keys)})"
        )


def parse_block(block_type: type[_T], raw: Any, path: str) -> _T:
    """Build dataclass ``block_type`` from the raw mapping of one configuration block.

    Field annotations select the converter from :data:`CONVERTERS` (see
    :func:`converter_for`); fields without a default are required; unknown
    keys are rejected. The dataclass's own ``__post_init__`` performs any
    range validation and may raise :class:`ConfigError` itself.
    """
    if raw is None:
        raise ConfigError(f"block '{path}' is empty")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"block '{path}' must be a mapping, got {type(raw).__name__}")
    specs = fields(block_type)  # type: ignore[arg-type]
    reject_unknown(raw, [spec.name for spec in specs], f"'{path}'")
    kwargs: dict[str, Any] = {}
    for spec in specs:
        key = f"{path}.{spec.name}"
        converter = converter_for(spec, key)
        if spec.name in raw:
            kwargs[spec.name] = converter(raw[spec.name], key)
        elif spec.default is not MISSING:
            kwargs[spec.name] = spec.default
        elif spec.default_factory is not MISSING:
            kwargs[spec.name] = spec.default_factory()
        else:
            raise ConfigError(f"missing required key '{key}'")
    return block_type(**kwargs)


def parse_yaml(text: str, source: str) -> dict[str, Any]:
    """Parse YAML text whose top level must be a non-empty mapping."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{source}: document is empty")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: top level must be a mapping, got {type(data).__name__}")
    return dict(data)

"""Studio AI Brain.

FastAPI service that turns a text prompt into a MIDI execution payload
using the baseline device map in config/devices.json.

The planner is deterministic. It does not call a language model and it
does not open a MIDI port. Provider keys, the active UI theme, and the
desktop executor address live in config/app_settings.json. PDF manuals
dropped in manuals/ are indexed by rag_processor.py. An inference request
reads the encrypted xAI key from that file and calls api.x.ai/v1.

Run it from any working directory:

    python brain_service.py

The process listens on 127.0.0.1:8000 unless STUDIO_AI_HOST or
STUDIO_AI_PORT is set. An optional .env file in the project root is
loaded with python-dotenv. STUDIO_AI_DEVICES_PATH can replace the map.
STUDIO_AI_SETTINGS_PATH, STUDIO_AI_MANUALS_PATH, and
STUDIO_AI_RAG_INDEX_PATH can replace those locations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag_processor import (
    IngestReport,
    ManualNotFoundError,
    ManualParseError,
    RagIndexError,
    index_chunk_count,
    index_display_path,
    ingest_manuals,
    manuals_directory as indexed_manuals_directory,
    validate_manual_filename,
)
from settings_store import (
    ProviderKeyDecryptError,
    ProviderKeyMissingError,
    SettingsError,
    SettingsUpdate,
    SettingsView,
    UiTheme,
    apply_update,
    load_settings,
    public_view,
    settings_display_path,
)
from xai_client import InferenceUpstreamError, complete_chat

if sys.version_info < (3, 10):
    raise RuntimeError("studio-ai-brain requires Python 3.10 or newer")

SERVICE_NAME = "studio-ai-brain"
SERVICE_VERSION = "1.1.0"
PROJECT_ROOT = Path(__file__).resolve().parent
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAP_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_NUMBER_PATTERN = re.compile(r"(?<![0-9])([0-9]{1,3})(?![0-9])")
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SECRET_FIELDS = frozenset({"xai_grok", "openai", "anthropic"})
_VALUE_WINDOW = 80
logger = logging.getLogger("studio_ai_brain")

MMC_COMMAND_BYTES: dict[str, int] = {
    "stop": 0x01,
    "play": 0x02,
    "deferred_play": 0x03,
    "fast_forward": 0x04,
    "rewind": 0x05,
    "record_strobe": 0x06,
    "record_exit": 0x07,
    "record_pause": 0x08,
    "pause": 0x09,
}

BindingAuthority = Literal[
    "mmc_specification",
    "manufacturer_documented",
    "midi_specification",
    "project_baseline",
]
DeviceCategory = Literal["daw", "synthesizer", "looper", "stage_piano"]
MmcCommand = Literal[
    "stop",
    "play",
    "deferred_play",
    "fast_forward",
    "rewind",
    "record_strobe",
    "record_exit",
    "record_pause",
    "pause",
]
ValueSource = Literal["prompt", "trigger"]
PlanStatus = Literal["planned", "no_actions_matched"]


class MapModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _required_text(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("must not be blank")
    return cleaned


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _required_text(value)


def _normalize_aliases(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(value.split()).casefold()
        if len(item) < 2:
            raise ValueError(f"alias {item!r} is too short")
        if item in seen:
            raise ValueError(f"duplicate alias {item!r}")
        seen.add(item)
        cleaned.append(item)
    return cleaned


def _check_ids(keys: Iterable[str], label: str) -> None:
    for key in keys:
        if _ID_PATTERN.fullmatch(key) is None:
            raise ValueError(f"{label} id {key!r} must match {_ID_PATTERN.pattern}")


def mmc_sysex(mmc_device_id: int, command_byte: int) -> list[int]:
    """Build an MMC command: F0 7F <device-id> 06 <command> F7."""
    return [0xF0, 0x7F, mmc_device_id, 0x06, command_byte, 0xF7]


class ConnectionSpec(MapModel):
    protocol: Literal["midi"]
    port_hint: str = Field(min_length=1)
    default_channel: int = Field(ge=1, le=16)

    @field_validator("port_hint")
    @classmethod
    def _port_hint(cls, value: str) -> str:
        return _required_text(value)


class ControlChangeMessage(MapModel):
    type: Literal["control_change"]
    channel: int = Field(ge=1, le=16)
    controller: int = Field(ge=0, le=127)
    mode: Literal["absolute", "momentary"]
    minimum: int = Field(default=0, ge=0, le=127)
    maximum: int = Field(default=127, ge=0, le=127)
    trigger_value: int | None = Field(default=None, ge=0, le=127)

    @model_validator(mode="after")
    def _ranges(self) -> ControlChangeMessage:
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.mode == "momentary":
            if self.trigger_value is None:
                raise ValueError("momentary controls require trigger_value")
            if not self.minimum <= self.trigger_value <= self.maximum:
                raise ValueError("trigger_value is outside minimum/maximum")
        elif self.trigger_value is not None:
            raise ValueError("absolute controls cannot set trigger_value")
        return self


class MmcMessage(MapModel):
    type: Literal["mmc"]
    mmc_device_id: int = Field(ge=0, le=127)
    command: MmcCommand
    command_byte: int = Field(ge=0, le=127)

    @model_validator(mode="after")
    def _command_byte(self) -> MmcMessage:
        expected = MMC_COMMAND_BYTES[self.command]
        if self.command_byte != expected:
            raise ValueError(
                f"MMC command {self.command} must use command_byte {expected}, got {self.command_byte}"
            )
        return self


MidiBinding = Annotated[ControlChangeMessage | MmcMessage, Field(discriminator="type")]


class ControlSpec(MapModel):
    label: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    binding_authority: BindingAuthority
    hardware_target: str = Field(min_length=1)
    message: MidiBinding
    notes: str | None = None

    @field_validator("label", "hardware_target")
    @classmethod
    def _labels(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, values: list[str]) -> list[str]:
        return _normalize_aliases(values)


class ControlGroup(MapModel):
    description: str = Field(min_length=1)
    fan_out: bool = False
    aliases: list[str] = Field(default_factory=list)
    controls: dict[str, ControlSpec]

    @field_validator("description")
    @classmethod
    def _description(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, values: list[str]) -> list[str]:
        return _normalize_aliases(values)

    @model_validator(mode="after")
    def _group_rules(self) -> ControlGroup:
        _check_ids(self.controls, "control")
        if not self.controls:
            raise ValueError("control group must contain at least one control")
        if self.fan_out and not self.aliases:
            raise ValueError("fan_out groups require at least one alias")
        return self


class DeviceSpec(MapModel):
    display_name: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    category: DeviceCategory
    enabled: bool = True
    aliases: list[str] = Field(min_length=1)
    connection: ConnectionSpec
    notes: str | None = None
    control_groups: dict[str, ControlGroup]

    @field_validator("display_name", "manufacturer", "model")
    @classmethod
    def _names(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, values: list[str]) -> list[str]:
        return _normalize_aliases(values)

    @model_validator(mode="after")
    def _device_rules(self) -> DeviceSpec:
        _check_ids(self.control_groups, "control group")
        if not self.control_groups:
            raise ValueError("device must contain at least one control group")
        seen_aliases: dict[str, str] = {}
        seen_cc: dict[tuple[int, int], str] = {}
        seen_mmc: dict[str, str] = {}

        def claim_alias(alias: str, owner: str) -> None:
            previous = seen_aliases.get(alias)
            if previous is not None:
                raise ValueError(f"alias {alias!r} is used by {previous} and {owner}")
            seen_aliases[alias] = owner

        for alias in self.aliases:
            claim_alias(alias, "device")
        for group_id, group in self.control_groups.items():
            for alias in group.aliases:
                claim_alias(alias, f"group {group_id}")
            for control_id, control in group.controls.items():
                owner = f"{group_id}.{control_id}"
                for alias in control.aliases:
                    claim_alias(alias, owner)
                message = control.message
                if isinstance(message, ControlChangeMessage):
                    key = (message.channel, message.controller)
                    if key in seen_cc:
                        raise ValueError(
                            f"CC {message.controller} on channel {message.channel} "
                            f"is used by {seen_cc[key]} and {owner}"
                        )
                    seen_cc[key] = owner
                else:
                    if message.command in seen_mmc:
                        raise ValueError(
                            f"MMC command {message.command} is used by {seen_mmc[message.command]} and {owner}"
                        )
                    seen_mmc[message.command] = owner
        return self


class DeviceMap(MapModel):
    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    map_id: str = Field(min_length=1)
    channel_numbering: Literal["display_1_to_16"]
    description: str = Field(min_length=1)
    devices: dict[str, DeviceSpec]

    @field_validator("map_id")
    @classmethod
    def _map_id(cls, value: str) -> str:
        cleaned = _required_text(value)
        if _MAP_ID_PATTERN.fullmatch(cleaned) is None:
            raise ValueError(f"map_id {cleaned!r} must match {_MAP_ID_PATTERN.pattern}")
        return cleaned

    @field_validator("description")
    @classmethod
    def _description(cls, value: str) -> str:
        return _required_text(value)

    @model_validator(mode="after")
    def _devices_present(self) -> DeviceMap:
        _check_ids(self.devices, "device")
        if not self.devices:
            raise ValueError("devices must not be empty")
        return self


class OrchestrateRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=4000)
    target_device_ids: list[str] = Field(default_factory=list, max_length=32)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("prompt", "session_id")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("target_device_ids")
    @classmethod
    def _device_ids(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("target_device_ids must not contain blank ids")
        if any(len(value) > 64 for value in cleaned):
            raise ValueError("target_device_ids entries must be 64 characters or fewer")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("target_device_ids must be unique")
        return cleaned


class ResolvedControlChange(ApiModel):
    type: Literal["control_change"]
    channel: int = Field(ge=1, le=16)
    controller: int = Field(ge=0, le=127)
    value: int = Field(ge=0, le=127)


class ResolvedMmc(ApiModel):
    type: Literal["mmc"]
    mmc_device_id: int = Field(ge=0, le=127)
    command: MmcCommand
    command_byte: int = Field(ge=0, le=127)
    sysex: list[int] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _sysex(self) -> ResolvedMmc:
        expected = mmc_sysex(self.mmc_device_id, self.command_byte)
        if self.sysex != expected:
            raise ValueError("sysex does not match the MMC command")
        return self


ResolvedMidi = Annotated[ResolvedControlChange | ResolvedMmc, Field(discriminator="type")]


class ExecutionStep(ApiModel):
    step_index: int = Field(ge=0)
    device_id: str
    port_hint: str
    control_group: str
    control_id: str
    label: str
    hardware_target: str
    binding_authority: BindingAuthority
    matched_alias: str
    value_source: ValueSource
    midi: ResolvedMidi


class ExecutionPayload(ApiModel):
    service: Literal["studio-ai-brain"] = SERVICE_NAME
    service_version: str
    planner: Literal["deterministic_mock"] = "deterministic_mock"
    schema_version: str
    map_id: str
    map_path: str
    channel_numbering: Literal["display_1_to_16"]
    request_id: str
    generated_at: str
    session_id: str | None
    status: PlanStatus
    prompt: str
    available_device_ids: list[str]
    disabled_device_ids: list[str]
    referenced_device_ids: list[str]
    steps: list[ExecutionStep]
    warnings: list[str]


class HealthResponse(ApiModel):
    status: Literal["ok"]
    service: Literal["studio-ai-brain"]
    service_version: str
    map_id: str
    schema_version: str
    map_path: str
    device_count: int = Field(ge=0)
    manuals_path: str
    manuals_present: bool
    ui_theme: UiTheme
    settings_path: str
    rag_index_path: str
    rag_chunk_count: int = Field(ge=0)


class ManualSource(ApiModel):
    source: str
    page: int = Field(ge=1)
    score: float = Field(ge=0, le=1)


class InferenceRequest(ApiModel):
    prompt: str = Field(min_length=1, max_length=8000)
    model: str = Field(default="grok-4.7", max_length=64)
    use_manuals: bool = True

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        cleaned = value.strip()
        if _MODEL_PATTERN.fullmatch(cleaned) is None:
            raise ValueError("model must be a single model id")
        return cleaned


class InferenceResponse(ApiModel):
    request_id: str
    model: str
    content: str = Field(min_length=1, max_length=20000)
    sources: list[ManualSource]


class RagUploadRequest(ApiModel):
    filename: str | None = Field(default=None, max_length=180)

    @field_validator("filename")
    @classmethod
    def _filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_manual_filename(value)


class UnknownDeviceError(Exception):
    def __init__(self, unknown: list[str], available: list[str]) -> None:
        super().__init__(", ".join(unknown))
        self.unknown = unknown
        self.available = available


class DeviceRegistry:
    def __init__(self, device_map: DeviceMap, source_path: Path) -> None:
        self.device_map = device_map
        self.source_path = source_path

    @classmethod
    def load(cls, path: Path) -> DeviceRegistry:
        return cls(load_device_map(path), path)

    def map_path(self) -> str:
        return display_path(self.source_path)


@dataclass(frozen=True)
class ControlHit:
    device_id: str
    group_id: str
    control_id: str
    span: tuple[int, int]
    alias: str

    @property
    def specificity(self) -> int:
        return len(self.alias)


@dataclass(frozen=True)
class NumberHit:
    span: tuple[int, int]
    value: int


@dataclass(frozen=True)
class ScopeDecision:
    scope: list[str]
    warnings: list[str]
    enabled_mentioned: list[str]


@dataclass(frozen=True)
class ValueFailure:
    hit: ControlHit
    kind: Literal["missing", "range"]
    value: int | None
    minimum: int
    maximum: int


def bootstrap_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def manuals_directory() -> Path:
    return indexed_manuals_directory()


def default_devices_path() -> Path:
    return PROJECT_ROOT / "config" / "devices.json"


def resolve_devices_path() -> Path:
    override = os.getenv("STUDIO_AI_DEVICES_PATH", "").strip()
    if not override:
        return default_devices_path()
    path = Path(override).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_device_map(path: Path) -> DeviceMap:
    if not path.is_file():
        raise FileNotFoundError(f"Device map not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Device map is not valid JSON: {path} ({exc})") from exc
    try:
        return DeviceMap.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Device map failed validation: {path}\n{exc}") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


@lru_cache(maxsize=512)
def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")


def alias_span(text: str, alias: str) -> tuple[int, int] | None:
    match = _alias_pattern(alias).search(text)
    if match is None:
        return None
    return match.start(), match.end()


def best_alias(text: str, aliases: Sequence[str]) -> tuple[str, tuple[int, int]] | None:
    winner_rank: tuple[int, int] | None = None
    winner: tuple[str, tuple[int, int]] | None = None
    for alias in aliases:
        span = alias_span(text, alias)
        if span is None:
            continue
        rank = (len(alias), -span[0])
        if winner_rank is None or rank > winner_rank:
            winner_rank = rank
            winner = (alias, span)
    return winner


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer != inner


def gap(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def mentioned_devices(text: str, devices: dict[str, DeviceSpec]) -> tuple[list[str], list[tuple[int, int]]]:
    names: list[str] = []
    spans: list[tuple[int, int]] = []
    for device_id, device in devices.items():
        found = best_alias(text, device.aliases)
        if found is None:
            continue
        _alias, span = found
        names.append(device_id)
        spans.append(span)
    return names, spans


def resolve_scope(
    devices: dict[str, DeviceSpec],
    mentioned: Sequence[str],
    targets: Sequence[str],
) -> ScopeDecision:
    warnings: list[str] = []
    unknown = [device_id for device_id in targets if device_id not in devices]
    if unknown:
        raise UnknownDeviceError(unknown, list(devices))

    disabled_seen: set[str] = set()
    for device_id in [*targets, *mentioned]:
        device = devices.get(device_id)
        if device is None or device.enabled or device_id in disabled_seen:
            continue
        disabled_seen.add(device_id)
        warnings.append(f"{device_id} is disabled and was skipped")

    enabled_targets = [device_id for device_id in targets if devices[device_id].enabled]
    enabled_mentioned = [device_id for device_id in mentioned if devices[device_id].enabled]
    if targets:
        scope = enabled_targets
        excluded = [device_id for device_id in enabled_mentioned if device_id not in scope]
        if len(excluded) == 1:
            warnings.append(f"{excluded[0]} was named in the prompt but is outside target_device_ids")
        elif excluded:
            warnings.append(f"{', '.join(excluded)} were named in the prompt but are outside target_device_ids")
    else:
        scope = [device_id for device_id, device in devices.items() if device.enabled]

    return ScopeDecision(scope, warnings, enabled_mentioned)


def suppress_shorter_hits(hits: Sequence[ControlHit]) -> list[ControlHit]:
    kept: list[ControlHit] = []
    for index, hit in enumerate(hits):
        dominated = False
        for other_index, other in enumerate(hits):
            if other_index == index or other.specificity <= hit.specificity:
                continue
            same_device = other.device_id == hit.device_id
            if span_contains(other.span, hit.span) or (same_device and spans_overlap(other.span, hit.span)):
                dominated = True
                break
        if not dominated:
            kept.append(hit)
    return kept


def collect_hits(text: str, devices: dict[str, DeviceSpec], scope: Sequence[str]) -> list[ControlHit]:
    individuals: list[ControlHit] = []
    scope_set = set(scope)
    for device_id, device in devices.items():
        if device_id not in scope_set:
            continue
        for group_id, group in device.control_groups.items():
            for control_id, control in group.controls.items():
                found = best_alias(text, control.aliases)
                if found is None:
                    continue
                alias, span = found
                individuals.append(ControlHit(device_id, group_id, control_id, span, alias))

    kept = suppress_shorter_hits(individuals)
    groups_with_hits = {(hit.device_id, hit.group_id) for hit in kept}
    selected = list(kept)
    for device_id in scope:
        device = devices[device_id]
        for group_id, group in device.control_groups.items():
            if (device_id, group_id) in groups_with_hits or not group.fan_out:
                continue
            found = best_alias(text, group.aliases)
            if found is None:
                continue
            alias, span = found
            for control_id in group.controls:
                selected.append(ControlHit(device_id, group_id, control_id, span, alias))
    return selected


def sort_hits(hits: Sequence[ControlHit], devices: dict[str, DeviceSpec]) -> list[ControlHit]:
    device_order = {device_id: index for index, device_id in enumerate(devices)}
    group_order: dict[tuple[str, str], int] = {}
    control_order: dict[tuple[str, str, str], int] = {}
    for device_id, device in devices.items():
        for group_index, (group_id, group) in enumerate(device.control_groups.items()):
            group_order[(device_id, group_id)] = group_index
            for control_index, control_id in enumerate(group.controls):
                control_order[(device_id, group_id, control_id)] = control_index
    return sorted(
        hits,
        key=lambda hit: (
            hit.span[0],
            hit.span[1],
            device_order[hit.device_id],
            group_order[(hit.device_id, hit.group_id)],
            control_order[(hit.device_id, hit.group_id, hit.control_id)],
        ),
    )


def extract_numbers(text: str, blocked: Sequence[tuple[int, int]]) -> list[NumberHit]:
    numbers: list[NumberHit] = []
    for match in _NUMBER_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if any(spans_overlap(span, block) for block in blocked):
            continue
        numbers.append(NumberHit(span, int(match.group(1))))
    return numbers


def assign_values(mentions: Sequence[tuple[int, int]], numbers: Sequence[NumberHit]) -> dict[int, int]:
    assigned: dict[int, int] = {}
    if not mentions or not numbers:
        return assigned
    if len(numbers) == 1:
        number = numbers[0]
        for index, span in enumerate(mentions):
            if gap(span, number.span) <= _VALUE_WINDOW:
                assigned[index] = number.value
        return assigned

    pairs: list[tuple[int, int, int, int]] = []
    for mention_index, span in enumerate(mentions):
        for number_index, number in enumerate(numbers):
            distance = gap(span, number.span)
            if distance > _VALUE_WINDOW:
                continue
            follows_penalty = 0 if number.span[0] >= span[1] else 1
            pairs.append((distance, follows_penalty, mention_index, number_index))
    pairs.sort()
    used_mentions: set[int] = set()
    used_numbers: set[int] = set()
    for _distance, _follows_penalty, mention_index, number_index in pairs:
        if mention_index in used_mentions or number_index in used_numbers:
            continue
        used_mentions.add(mention_index)
        used_numbers.add(number_index)
        assigned[mention_index] = numbers[number_index].value
    return assigned


def _is_absolute(control: ControlSpec) -> bool:
    message = control.message
    return isinstance(message, ControlChangeMessage) and message.mode == "absolute"


def failure_warnings(failures: Sequence[ValueFailure]) -> list[str]:
    grouped: dict[tuple[str, str, tuple[int, int], str, int | None, int, int], list[str]] = {}
    order: list[tuple[str, str, tuple[int, int], str, int | None, int, int]] = []
    for failure in failures:
        key = (
            failure.hit.device_id,
            failure.hit.group_id,
            failure.hit.span,
            failure.kind,
            failure.value,
            failure.minimum,
            failure.maximum,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(failure.hit.control_id)

    messages: list[str] = []
    for key in order:
        device_id, group_id, _span, kind, value, minimum, maximum = key
        names = ", ".join(grouped[key])
        if kind == "missing":
            messages.append(
                f"{device_id}.{group_id} ({names}) matched but no value between {minimum} and {maximum} was found"
            )
        else:
            messages.append(f"{device_id}.{group_id} ({names}) value {value} is outside {minimum}-{maximum}")
    return messages


def build_steps(
    hits: Sequence[ControlHit],
    devices: dict[str, DeviceSpec],
    assigned: dict[int, int],
    span_to_mention: dict[tuple[int, int], int],
) -> tuple[list[ExecutionStep], list[str]]:
    steps: list[ExecutionStep] = []
    failures: list[ValueFailure] = []
    for hit in hits:
        device = devices[hit.device_id]
        control = device.control_groups[hit.group_id].controls[hit.control_id]
        message = control.message
        if isinstance(message, MmcMessage):
            midi: ResolvedMidi = ResolvedMmc(
                type="mmc",
                mmc_device_id=message.mmc_device_id,
                command=message.command,
                command_byte=message.command_byte,
                sysex=mmc_sysex(message.mmc_device_id, message.command_byte),
            )
            value_source: ValueSource = "trigger"
        elif message.mode == "momentary":
            if message.trigger_value is None:
                raise RuntimeError(f"{hit.device_id}.{hit.group_id}.{hit.control_id} is missing trigger_value")
            midi = ResolvedControlChange(
                type="control_change",
                channel=message.channel,
                controller=message.controller,
                value=message.trigger_value,
            )
            value_source = "trigger"
        else:
            value = assigned.get(span_to_mention[hit.span])
            if value is None or not message.minimum <= value <= message.maximum:
                failures.append(
                    ValueFailure(
                        hit=hit,
                        kind="missing" if value is None else "range",
                        value=value,
                        minimum=message.minimum,
                        maximum=message.maximum,
                    )
                )
                continue
            midi = ResolvedControlChange(
                type="control_change",
                channel=message.channel,
                controller=message.controller,
                value=value,
            )
            value_source = "prompt"

        steps.append(
            ExecutionStep(
                step_index=len(steps),
                device_id=hit.device_id,
                port_hint=device.connection.port_hint,
                control_group=hit.group_id,
                control_id=hit.control_id,
                label=control.label,
                hardware_target=control.hardware_target,
                binding_authority=control.binding_authority,
                matched_alias=hit.alias,
                value_source=value_source,
                midi=midi,
            )
        )
    return steps, failure_warnings(failures)


def plan_execution(request: OrchestrateRequest, registry: DeviceRegistry) -> ExecutionPayload:
    device_map = registry.device_map
    devices = device_map.devices
    text = normalize_text(request.prompt)
    mentioned, device_spans = mentioned_devices(text, devices)
    decision = resolve_scope(devices, mentioned, request.target_device_ids)
    hits = sort_hits(collect_hits(text, devices, decision.scope), devices)
    warnings = list(decision.warnings)
    if decision.enabled_mentioned:
        allowed = set(decision.enabled_mentioned).intersection(decision.scope)
        kept: list[ControlHit] = []
        dropped: dict[str, list[str]] = {}
        for hit in hits:
            if hit.device_id in allowed:
                kept.append(hit)
                continue
            control_ids = dropped.setdefault(hit.device_id, [])
            if hit.control_id not in control_ids:
                control_ids.append(hit.control_id)
        for device_id, control_ids in dropped.items():
            warnings.append(
                f"{device_id} ({', '.join(control_ids)}) matched but was not named in the prompt"
            )
        hits = kept

    span_to_mention: dict[tuple[int, int], int] = {}
    mentions: list[tuple[int, int]] = []
    for hit in hits:
        control = devices[hit.device_id].control_groups[hit.group_id].controls[hit.control_id]
        if not _is_absolute(control) or hit.span in span_to_mention:
            continue
        span_to_mention[hit.span] = len(mentions)
        mentions.append(hit.span)

    blocked = [*device_spans, *(hit.span for hit in hits)]
    assigned = assign_values(mentions, extract_numbers(text, blocked))
    steps, value_warnings = build_steps(hits, devices, assigned, span_to_mention)

    hit_devices = {hit.device_id for hit in hits}
    for device_id in decision.enabled_mentioned:
        if device_id in decision.scope and device_id not in hit_devices:
            warnings.append(f"{device_id} was named in the prompt but no mapped control matched")
    warnings.extend(value_warnings)

    referenced: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step.device_id not in seen:
            seen.add(step.device_id)
            referenced.append(step.device_id)

    return ExecutionPayload(
        service_version=SERVICE_VERSION,
        schema_version=device_map.schema_version,
        map_id=device_map.map_id,
        map_path=registry.map_path(),
        channel_numbering=device_map.channel_numbering,
        request_id=str(uuid.uuid4()),
        generated_at=utc_now(),
        session_id=request.session_id,
        status="planned" if steps else "no_actions_matched",
        prompt=request.prompt,
        available_device_ids=list(devices),
        disabled_device_ids=[device_id for device_id, device in devices.items() if not device.enabled],
        referenced_device_ids=referenced,
        steps=steps,
        warnings=warnings,
    )


def _redact_secrets(value: object) -> object:
    if isinstance(value, dict):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            if key in _SECRET_FIELDS:
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _validation_response(exc: RequestValidationError) -> JSONResponse:
    details: list[object] = []
    for error in exc.errors():
        item = dict(error)
        location = item.get("loc", ())
        secret_field = any(isinstance(part, str) and part in _SECRET_FIELDS for part in location)
        if "input" in item:
            item["input"] = "[redacted]" if secret_field else _redact_secrets(item["input"])
        context = item.get("ctx")
        if isinstance(context, dict):
            item["ctx"] = {
                key: str(value) if isinstance(value, BaseException) else value
                for key, value in context.items()
            }
        details.append(item)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(details)})


def create_app() -> FastAPI:
    bootstrap_env()
    registry = DeviceRegistry.load(resolve_devices_path())
    logger.info(
        "loaded %s schema %s with %d devices from %s",
        registry.device_map.map_id,
        registry.device_map.schema_version,
        len(registry.device_map.devices),
        registry.map_path(),
    )
    app = FastAPI(
        title="Studio AI Brain",
        version=SERVICE_VERSION,
        summary="Plans multi-device MIDI execution payloads from a text prompt.",
        description=(
            "Background orchestration service. The deterministic planner reads the "
            "device map and does not call an external language model or open a MIDI port. "
            "Provider keys and the active theme are stored in config/app_settings.json. "
            "Gear manuals dropped in manuals/ are indexed into data/rag_index.json. "
            "Inference calls to api.x.ai/v1 load the encrypted xAI key on each request."
        ),
    )
    app.state.registry = registry

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _validation_response(exc)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        device_map = registry.device_map
        manuals = manuals_directory()
        try:
            theme = load_settings().ui_theme
        except SettingsError:
            logger.warning("app settings are unavailable")
            theme = "dark_studio"
        try:
            chunk_count = index_chunk_count()
        except RagIndexError:
            logger.warning("rag index is unavailable")
            chunk_count = 0
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            service_version=SERVICE_VERSION,
            map_id=device_map.map_id,
            schema_version=device_map.schema_version,
            map_path=registry.map_path(),
            device_count=len(device_map.devices),
            manuals_path=display_path(manuals),
            manuals_present=manuals.is_dir(),
            ui_theme=theme,
            settings_path=settings_display_path(),
            rag_index_path=index_display_path(),
            rag_chunk_count=chunk_count,
        )

    @app.get("/v1/settings", response_model=SettingsView)
    def read_settings() -> SettingsView:
        try:
            return public_view(load_settings())
        except SettingsError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

    @app.post("/v1/settings/update", response_model=SettingsView)
    def update_settings(body: SettingsUpdate) -> SettingsView:
        try:
            view = apply_update(body)
        except SettingsError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
        logger.info("settings updated theme=%s", view.ui_theme)
        return view

    @app.post("/v1/rag/upload", response_model=IngestReport)
    def rag_upload(body: RagUploadRequest | None = None) -> IngestReport:
        request = body if body is not None else RagUploadRequest()
        try:
            report = ingest_manuals(filename=request.filename)
        except ManualNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": "Manual was not found in manuals/.", "filename": exc.filename},
            ) from exc
        except ManualParseError as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": "Manual could not be indexed.", "filename": exc.filename, "reason": exc.reason},
            ) from exc
        except RagIndexError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
        logger.info(
            "rag upload indexed=%d unchanged=%d removed=%d errors=%d chunks=%d",
            len(report.indexed),
            len(report.unchanged),
            len(report.removed),
            len(report.errors),
            report.chunk_count,
        )
        return report

    @app.post("/v1/infer", response_model=InferenceResponse)
    def infer(body: InferenceRequest) -> InferenceResponse:
        request_id = str(uuid.uuid4())
        try:
            result = complete_chat(
                prompt=body.prompt,
                model=body.model,
                use_manuals=body.use_manuals,
                user_agent=f"{SERVICE_NAME}/{SERVICE_VERSION}",
            )
        except ProviderKeyMissingError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
        except ProviderKeyDecryptError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
        except InferenceUpstreamError as exc:
            logger.warning("infer request_id=%s failed", request_id)
            raise HTTPException(status_code=502, detail={"message": str(exc)}) from exc
        except RagIndexError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
        logger.info("infer request_id=%s model=%s sources=%d", request_id, body.model, len(result.sources))
        return InferenceResponse(
            request_id=request_id,
            model=body.model,
            content=result.content,
            sources=[
                ManualSource(
                    source=hit.source,
                    page=hit.page,
                    score=min(1.0, max(0.0, round(hit.score, 4))),
                )
                for hit in result.sources
            ],
        )

    @app.post("/v1/orchestrate", response_model=ExecutionPayload)
    def orchestrate(body: OrchestrateRequest) -> ExecutionPayload:
        try:
            payload = plan_execution(body, registry)
        except UnknownDeviceError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unknown target_device_ids.",
                    "unknown_device_ids": exc.unknown,
                    "available_device_ids": exc.available,
                },
            ) from exc
        logger.info(
            "orchestrate request_id=%s status=%s steps=%d warnings=%d",
            payload.request_id,
            payload.status,
            len(payload.steps),
            len(payload.warnings),
        )
        return payload

    # ui-console calls these routes from the Tauri webview, a separate local origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?"
            r"|https?://tauri\.localhost"
            r"|https?://asset\.localhost"
            r"|tauri://localhost)$"
        ),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
        allow_credentials=False,
        allow_private_network=True,
    )

    return app


def resolve_port(raw: str | None) -> int:
    if raw is None or not raw.strip():
        return 8000
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"STUDIO_AI_PORT must be an integer, got {raw!r}") from exc
    if port < 1 or port > 65535:
        raise SystemExit("STUDIO_AI_PORT must be between 1 and 65535")
    return port


def main() -> None:
    bootstrap_env()
    host = os.getenv("STUDIO_AI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = resolve_port(os.getenv("STUDIO_AI_PORT"))
    uvicorn.run(app, host=host, port=port)


app = create_app()


if __name__ == "__main__":
    main()

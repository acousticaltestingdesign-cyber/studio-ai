"""Bring-your-own-key settings for studio-ai-brain.

config/app_settings.json stores the active theme, the desktop executor
address, and provider keys. Saved keys are Fernet ciphertext. The Fernet
key lives in the OS user-data directory, outside this project, and each
inference call decrypts only the key it needs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from ipaddress import IPv4Address
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger("studio_ai_brain.settings")

PROJECT_ROOT = Path(__file__).resolve().parent
UiTheme = Literal["dark_studio", "retro_analog", "cyberpunk_synth"]
ProviderName = Literal["xai_grok", "openai", "anthropic"]
_PROVIDER_NAMES: tuple[ProviderName, ...] = ("xai_grok", "openai", "anthropic")
_LOCK = threading.Lock()


class SettingsError(Exception):
    """Base error for settings that cannot be read, written, or decrypted."""


class SettingsCorruptError(SettingsError):
    def __init__(self) -> None:
        super().__init__("App settings could not be read.")


class SettingsWriteError(SettingsError):
    def __init__(self) -> None:
        super().__init__("App settings could not be saved.")


class SettingsKeyError(SettingsError):
    def __init__(self) -> None:
        super().__init__("The local settings key could not be read.")


class ProviderKeyMissingError(SettingsError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"The {provider} key is not configured.")
        self.provider = provider


class ProviderKeyDecryptError(SettingsError):
    def __init__(self, provider: str) -> None:
        super().__init__(f"The saved {provider} key could not be decrypted. Save it again.")
        self.provider = provider


class ProviderKeys(BaseModel):
    model_config = ConfigDict(extra="forbid")
    xai_grok: str = ""
    openai: str = ""
    anthropic: str = ""

    @field_validator("xai_grok", "openai", "anthropic")
    @classmethod
    def _stored_secret(cls, value: str) -> str:
        if value == "":
            return ""
        if len(value) > 4096 or any(ord(character) < 33 or ord(character) > 126 for character in value):
            raise ValueError("stored provider key must be empty or printable ciphertext")
        return value


class NetworkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desktop_executor_ip: str
    port: int = Field(ge=1, le=65535)

    @field_validator("desktop_executor_ip")
    @classmethod
    def _ipv4(cls, value: str) -> str:
        try:
            return str(IPv4Address(value.strip()))
        except ValueError as exc:
            raise ValueError("desktop_executor_ip must be an IPv4 address") from exc


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_keys: ProviderKeys
    ui_theme: UiTheme
    network: NetworkSettings


class ProviderKeyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    xai_grok: str | None = None
    openai: str | None = None
    anthropic: str | None = None

    @field_validator("xai_grok", "openai", "anthropic")
    @classmethod
    def _incoming_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            return ""
        cleaned = value.strip()
        printable = all(32 < ord(character) < 127 for character in cleaned)
        if cleaned == "" or len(cleaned) > 2048 or not printable:
            raise ValueError("provider key must be printable and at most 2048 characters")
        return cleaned


class NetworkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    desktop_executor_ip: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("desktop_executor_ip")
    @classmethod
    def _ipv4(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(IPv4Address(value.strip()))
        except ValueError as exc:
            raise ValueError("desktop_executor_ip must be an IPv4 address") from exc

    @model_validator(mode="after")
    def _present(self) -> NetworkInput:
        if self.desktop_executor_ip is None and self.port is None:
            raise ValueError("network update must include an address or a port")
        return self


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_keys: ProviderKeyInput | None = None
    ui_theme: UiTheme | None = None
    network: NetworkInput | None = None

    @model_validator(mode="after")
    def _present(self) -> SettingsUpdate:
        if self.provider_keys is None and self.ui_theme is None and self.network is None:
            raise ValueError("update must include provider_keys, ui_theme, or network")
        if self.provider_keys is not None and all(getattr(self.provider_keys, name) is None for name in _PROVIDER_NAMES):
            raise ValueError("provider_keys must include at least one provider")
        return self


class ProviderKeyStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    configured: bool


class ProviderKeyStatuses(BaseModel):
    model_config = ConfigDict(extra="forbid")
    xai_grok: ProviderKeyStatus
    openai: ProviderKeyStatus
    anthropic: ProviderKeyStatus


class SettingsView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ui_theme: UiTheme
    provider_keys: ProviderKeyStatuses
    network: NetworkSettings


def settings_path() -> Path:
    return _resolve_path("STUDIO_AI_SETTINGS_PATH", PROJECT_ROOT / "config" / "app_settings.json")


def settings_display_path() -> str:
    return _display(settings_path())


def default_settings() -> AppSettings:
    return AppSettings(
        provider_keys=ProviderKeys(),
        ui_theme="dark_studio",
        network=NetworkSettings(desktop_executor_ip="192.168.0.103", port=8080),
    )


def load_settings() -> AppSettings:
    with _LOCK:
        return _read_unlocked()


def public_view(settings: AppSettings) -> SettingsView:
    keys = settings.provider_keys
    return SettingsView(
        ui_theme=settings.ui_theme,
        provider_keys=ProviderKeyStatuses(
            xai_grok=ProviderKeyStatus(configured=bool(keys.xai_grok)),
            openai=ProviderKeyStatus(configured=bool(keys.openai)),
            anthropic=ProviderKeyStatus(configured=bool(keys.anthropic)),
        ),
        network=settings.network,
    )


def apply_update(patch: SettingsUpdate) -> SettingsView:
    with _LOCK:
        updated = _merge(_read_unlocked(), patch)
        _write_unlocked(updated)
        return public_view(updated)


def decrypt_provider_key(provider: ProviderName) -> str:
    with _LOCK:
        stored = getattr(_read_unlocked().provider_keys, provider)
        if stored == "":
            return ""
        cipher = _load_fernet()
    try:
        plain = cipher.decrypt(stored.encode("ascii"))
        text = plain.decode("utf-8")
    except (InvalidToken, UnicodeError, ValueError) as exc:
        raise ProviderKeyDecryptError(provider) from exc
    if text == "" or any(ord(character) < 32 or ord(character) > 126 for character in text):
        raise ProviderKeyDecryptError(provider)
    return text


def require_provider_key(provider: ProviderName) -> str:
    secret = decrypt_provider_key(provider)
    if secret == "":
        raise ProviderKeyMissingError(provider)
    return secret


def _merge(current: AppSettings, patch: SettingsUpdate) -> AppSettings:
    keys = current.provider_keys.model_dump()
    if patch.provider_keys is not None:
        for name in _PROVIDER_NAMES:
            incoming = getattr(patch.provider_keys, name)
            if incoming is None:
                continue
            keys[name] = "" if incoming == "" else _encrypt(incoming)
    network = current.network.model_dump()
    if patch.network is not None:
        if patch.network.desktop_executor_ip is not None:
            network["desktop_executor_ip"] = patch.network.desktop_executor_ip
        if patch.network.port is not None:
            network["port"] = patch.network.port
    theme = current.ui_theme if patch.ui_theme is None else patch.ui_theme
    return AppSettings.model_validate({"provider_keys": keys, "ui_theme": theme, "network": network})


def _encrypt(secret: str) -> str:
    return _load_fernet().encrypt(secret.encode("utf-8")).decode("ascii")


def _read_unlocked() -> AppSettings:
    path = settings_path()
    if not path.is_file():
        defaults = default_settings()
        _write_unlocked(defaults)
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("app settings file is unreadable: %s", exc.__class__.__name__)
        raise SettingsCorruptError() from exc
    try:
        return AppSettings.model_validate(raw)
    except ValidationError as exc:
        logger.warning("app settings failed validation fields=%s", _error_fields(exc))
        raise SettingsCorruptError() from exc


def _write_unlocked(settings: AppSettings) -> None:
    path = settings_path()
    try:
        _atomic_write(path, settings.model_dump_json(indent=2) + "\n")
        _tighten(path)
    except OSError as exc:
        logger.warning("app settings could not be saved: %s", exc.__class__.__name__)
        raise SettingsWriteError() from exc


def _load_fernet() -> Fernet:
    path = _encryption_key_path()
    if path.is_file():
        try:
            raw = path.read_bytes().strip()
            return Fernet(raw)
        except (OSError, ValueError, UnicodeError) as exc:
            logger.warning("settings key is unreadable: %s", exc.__class__.__name__)
            raise SettingsKeyError() from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = Fernet.generate_key()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            return Fernet(path.read_bytes().strip())
        except (OSError, ValueError, UnicodeError) as exc:
            raise SettingsKeyError() from exc
    try:
        os.write(descriptor, generated + b"\n")
    except OSError as exc:
        raise SettingsKeyError() from exc
    finally:
        os.close(descriptor)
    _tighten(path)
    return Fernet(generated)


def _encryption_key_path() -> Path:
    override = os.getenv("STUDIO_AI_SETTINGS_KEY_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _platform_data_dir() / "studio-ai" / "settings.key"


def _platform_data_dir() -> Path:
    if sys.platform == "win32":
        local = os.getenv("LOCALAPPDATA", "").strip()
        if local:
            return Path(local)
        return Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    data_home = os.getenv("XDG_DATA_HOME", "").strip()
    if data_home:
        return Path(data_home)
    return Path.home() / ".local" / "share"


def _resolve_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _tighten(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("could not restrict permissions on %s", path.name)


def _error_fields(exc: ValidationError) -> str:
    fields = sorted(".".join(str(part) for part in error["loc"]) for error in exc.errors())
    return ", ".join(fields)

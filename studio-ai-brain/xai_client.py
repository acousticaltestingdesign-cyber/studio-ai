"""xAI inference calls for studio-ai-brain.

Every call reads config/app_settings.json and decrypts the xai_grok key
at that moment, then posts to https://api.x.ai/v1/chat/completions.
Manual excerpts are attached only for that request. The key is not written
to the index and is not returned to the caller.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from rag_processor import ManualHit, search_manuals
from settings_store import require_provider_key

logger = logging.getLogger("studio_ai_brain.inference")

XAI_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
XAI_TIMEOUT_SECONDS = 120.0
_EXCERPT_CHARS = 800
_CONTEXT_CHARS = 4000
_SYSTEM_PROMPT = (
    "You are the Studio AI desk assistant. Answer from the supplied gear-manual "
    "excerpts when they apply. If the excerpts do not contain the answer, say so. "
    "Do not invent MIDI controller numbers, channels, or sysex bytes."
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class InferenceUpstreamError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class ChatResult:
    content: str
    sources: tuple[ManualHit, ...]


def complete_chat(*, prompt: str, model: str, use_manuals: bool, user_agent: str) -> ChatResult:
    secret = require_provider_key("xai_grok")
    try:
        context = ""
        sources: tuple[ManualHit, ...] = ()
        if use_manuals:
            context, sources = _manual_context(prompt)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(prompt, context)},
        ]
        content = _post_chat(secret, model=model, messages=messages, user_agent=user_agent)
        return ChatResult(content=content, sources=sources)
    finally:
        secret = ""


def _manual_context(prompt: str) -> tuple[str, tuple[ManualHit, ...]]:
    blocks: list[str] = []
    kept: list[ManualHit] = []
    used = 0
    for hit in search_manuals(prompt, limit=4):
        excerpt = hit.text.strip()
        if len(excerpt) > _EXCERPT_CHARS:
            excerpt = excerpt[:_EXCERPT_CHARS].rstrip() + "..."
        block = f"[{hit.source} page {hit.page}]\n{excerpt}"
        if blocks and used + len(block) > _CONTEXT_CHARS:
            break
        blocks.append(block)
        kept.append(hit)
        used += len(block)
    return "\n\n".join(blocks), tuple(kept)


def _user_content(prompt: str, context: str) -> str:
    if not context:
        return prompt
    return f"Manual excerpts:\n{context}\n\nQuestion:\n{prompt}"


def _post_chat(secret: str, *, model: str, messages: list[dict[str, str]], user_agent: str) -> str:
    header = f"Bearer {secret}"
    secret = ""
    payload = {"model": model, "messages": messages, "stream": False}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        XAI_CHAT_COMPLETIONS_URL,
        data=encoded,
        headers={
            "Authorization": header,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    header = ""
    try:
        return _read_completion(request)
    finally:
        if request.has_header("Authorization"):
            request.remove_header("Authorization")


def _read_completion(request: urllib.request.Request) -> str:
    try:
        with _OPENER.open(request, timeout=XAI_TIMEOUT_SECONDS) as response:
            raw = response.read(1_048_576)
    except TimeoutError as exc:
        raise InferenceUpstreamError("The xAI request timed out.") from exc
    except urllib.error.HTTPError as exc:
        try:
            exc.read(1024)
        except OSError:
            pass
        logger.warning("xAI request failed status=%s", exc.code)
        raise InferenceUpstreamError(f"xAI returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        logger.warning("xAI request failed error=%s", exc.__class__.__name__)
        raise InferenceUpstreamError("The xAI service could not be reached.") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceUpstreamError("xAI returned an unreadable response.") from exc
    return _completion_text(parsed)


def _completion_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise InferenceUpstreamError("xAI returned an unreadable response.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise InferenceUpstreamError("xAI returned an unreadable response.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise InferenceUpstreamError("xAI returned an unreadable response.")
    content = message.get("content")
    text = _content_string(content).strip()
    if not text:
        raise InferenceUpstreamError("xAI returned an empty completion.")
    return text[:20000]


def _content_string(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(part.strip() for part in parts if part.strip())

"""Rewrite the completed draft through the configured paraphrase API.

The API is a prose service and intentionally removes Markdown decoration from
its response.  This node therefore sends only visible text segments wrapped in
stable markers, then rebuilds the original Markdown/HTML template locally.
That keeps heading levels, inline styles, links, code and screenshot jobs out
of the rewrite service while still rewriting every visible prose segment.
"""

from __future__ import annotations

import re
import hashlib
import time
import threading
from dataclasses import dataclass

import requests
from agent.state import AgentState
from agent.config import get_config


DEFAULT_ENDPOINT = "https://rapi.ycjg.top/api/v2/paraphrase"
DEFAULT_PRESET = "aimove_matrix_s_1"
# The provider preserves markers for ordinary batches but may reorder them for
# particular long/rich passages. Keep batches small, then retry only a failed
# batch as individual plain-text requests.
# The successful production shape is a small marked batch. If the provider
# drops any marker, only that batch is recovered one segment at a time through
# tokenless visible-span requests; the rest of the article remains untouched.
MAX_BATCH_SEGMENTS = 8
MAX_BATCH_CHARS = 8_000
MIN_SEGMENT_CHARS = 96
# The paraphrase provider is unreliable for short, unframed requests. The
# failure is not limited to 3-8 characters: headings and list labels around
# 20-30 characters have also returned the same HTTP 500. Keep every short
# recovery request inside a two-span marked payload and extract only span 0.
SHORT_SEGMENT_MAX_CHARS = 96
TARGET_SEGMENT_CHARS = 900
REQUEST_TIMEOUT_SECONDS = 600
MAX_TRANSIENT_ATTEMPTS = 6
SEGMENT_ATTEMPTS = 2
SHORT_SEGMENT_ATTEMPTS = 3
MAX_OUTAGE_WAIT_SECONDS = 0
OUTAGE_POLL_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 0.6
RECOVERY_ATTEMPTS = 2
RECOVERY_DELAYS_SECONDS = (2, 5, 10)
_REQUEST_GATE = threading.Lock()
_LAST_REQUEST_AT = 0.0
_PARAPHRASE_RUN_LOCK = threading.Lock()
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
PARAPHRASE_ENGINE_VERSION = "paraphrase-v25-marker-leak-proof-20260819"

# These tokens are Markdown/HTML structure or workflow metadata. They remain
# in the template and are never sent as rewriteable text.
_SPECIAL = re.compile(
    r"(?:"
    r"\[SCREENSHOT:[^\]\r\n]*\]"
    r"|!\[[^\]\r\n]*\]\([^\)\r\n]+\)"
    r"|`[^`\r\n]*`"
    r"|https?://[^\s<>\]\)]+"
    r"|</?[^>\r\n]+>"
    r"|\]\([^\)\r\n]*\)"
    r"|[\[\]|]"
    r"|\*\*|__|~~"
    r"|(?<!\w)\*(?!\s)|(?<!\w)_(?!\w)"
    r")"
)
_PREFIX = re.compile(
    r"^(\s{0,3}(?:#{1,6}\s+|(?:>\s*)+|(?:[-+*]\s+|\d+[.)]\s+)))"
)
_FENCE = re.compile(r"^\s*(```|~~~)")
_MARKER = re.compile(r"CASEG(\d{6})START(.*?)CASEG\1END", re.DOTALL)
_ANY_MARKER_TOKEN = re.compile(r"CASEG\d{6}(?:START|END)")
_PROTECTED_TOKEN = re.compile(r"\[CAGKEEP\d{6}\]")
_TRAILING_NEWLINE = re.compile(r"(\r?\n)$")


@dataclass
class _RewritePlan:
    template: str
    texts: list[str]
    protected: dict[str, str]


class ParaphraseAPIError(RuntimeError):
    """API failure metadata that never includes article text or credentials."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _has_visible_text(value: str) -> bool:
    return bool(re.search(r"[\w\u3400-\u9fff]", value, re.UNICODE))


def _add_segment(parts: list[str], texts: list[str], value: str) -> None:
    """Add one editable segment while preserving its surrounding whitespace."""
    if not _has_visible_text(value):
        parts.append(value)
        return
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    core_end = len(value) - len(trailing) if trailing else len(value)
    core = value[len(leading) : core_end]
    if not core or not _has_visible_text(core):
        parts.append(value)
        return
    index = len(texts)
    texts.append(core)
    parts.extend((leading, f"CASEG{index:06d}START", core, f"CASEG{index:06d}END", trailing))


def _rewrite_line(
    line: str, texts: list[str], protected: dict[str, str]
) -> str:
    ending_match = _TRAILING_NEWLINE.search(line)
    ending = ending_match.group(1) if ending_match else ""
    body = line[: -len(ending)] if ending else line
    prefix_match = _PREFIX.match(body)
    prefix = prefix_match.group(1) if prefix_match else ""
    content = body[len(prefix) :]
    suffix = ""
    # ATX headings may use trailing hashes. Keep them outside the rewriteable
    # span so the API cannot turn a heading into a differently styled line.
    if re.match(r"^\s{0,3}#{1,6}\s+", prefix):
        suffix_match = re.search(r"(\s+#{1,6})\s*$", content)
        if suffix_match:
            suffix = suffix_match.group(1)
            content = content[: suffix_match.start()]
    # Keep one request unit per complete line/paragraph. Protected Markdown,
    # HTML and URL tokens become stable placeholders inside that unit; this
    # avoids turning one paragraph into many API calls while allowing exact
    # local restoration after the prose rewrite.
    masked: list[str] = []
    cursor = 0
    has_editable_text = False
    for match in _SPECIAL.finditer(content):
        editable = content[cursor : match.start()]
        masked.append(editable)
        has_editable_text = has_editable_text or _has_visible_text(editable)
        token = f"[CAGKEEP{len(protected):06d}]"
        protected[token] = match.group(0)
        masked.append(token)
        cursor = match.end()
    tail = content[cursor:]
    masked.append(tail)
    has_editable_text = has_editable_text or _has_visible_text(tail)
    parts: list[str] = [prefix]
    if has_editable_text:
        _add_segment(parts, texts, "".join(masked))
    else:
        parts.append(content)
    parts.append(suffix)
    parts.append(ending)
    return "".join(parts)


def _build_plan(draft: str) -> _RewritePlan:
    texts: list[str] = []
    protected: dict[str, str] = {}
    output: list[str] = []
    in_fence = False
    pending: list[str] = []
    pending_chars = 0

    def protect(value: str) -> str:
        token = f"[CAGKEEP{len(protected):06d}]"
        protected[token] = value
        return token

    def flush_pending() -> None:
        nonlocal pending_chars
        if not pending:
            return
        value = "".join(pending)
        _add_segment(output, texts, value)
        pending.clear()
        pending_chars = 0

    def queue_line(line: str) -> None:
        """Append a line to a semantic request unit with all syntax masked.

        The provider is unreliable for 3-5 character requests. A line-level
        plan therefore made an otherwise healthy batch fall back to tiny
        individual requests. Keep formatting, URLs and line breaks as stable
        placeholders, while sending adjacent visible prose as one unit.
        """
        nonlocal pending_chars
        ending_match = _TRAILING_NEWLINE.search(line)
        ending = ending_match.group(1) if ending_match else ""
        body = line[: -len(ending)] if ending else line
        prefix_match = _PREFIX.match(body)
        prefix = prefix_match.group(1) if prefix_match else ""
        content = body[len(prefix) :]
        suffix = ""
        if re.match(r"^\s{0,3}#{1,6}\s+", prefix):
            suffix_match = re.search(r"(\s+#{1,6})\s*$", content)
            if suffix_match:
                suffix = suffix_match.group(1)
                content = content[: suffix_match.start()]

        masked: list[str] = [protect(prefix)] if prefix else []
        cursor = 0
        has_editable_text = False
        for match in _SPECIAL.finditer(content):
            editable = content[cursor : match.start()]
            masked.append(editable)
            has_editable_text = has_editable_text or _has_visible_text(editable)
            masked.append(protect(match.group(0)))
            cursor = match.end()
        tail = content[cursor:]
        masked.append(tail)
        has_editable_text = has_editable_text or _has_visible_text(tail)
        if suffix:
            masked.append(protect(suffix))
        if ending:
            masked.append(protect(ending))

        # Syntax-only lines (images, screenshot jobs, separators) have no
        # rewriteable prose. Preserve them directly instead of making an API
        # request that cannot make a textual change.
        if not has_editable_text:
            # Keep blank-line structure inside a pending semantic unit so a
            # short heading can borrow context from the following paragraph.
            if not content.strip() and pending:
                pending.extend(masked)
                return
            flush_pending()
            output.append(line)
            return

        pending.extend(masked)
        pending_chars += len(content)
        # Target-sized units retain enough context for a meaningful rewrite.
        # A short heading/list line remains queued for the following prose.
        if pending_chars >= TARGET_SEGMENT_CHARS:
            flush_pending()

    for line in draft.splitlines(keepends=True):
        fence = _FENCE.match(line)
        if fence:
            flush_pending()
            in_fence = not in_fence
            output.append(line)
        elif in_fence:
            flush_pending()
            output.append(line)
        else:
            queue_line(line)
            # A blank line is already represented by a protected newline in
            # the current unit. Once enough context exists, it is a stable
            # semantic boundary at which to send the request.
            if not line.strip() and pending_chars >= MIN_SEGMENT_CHARS:
                flush_pending()
    flush_pending()
    return _RewritePlan("".join(output), texts, protected)


def _request(
    text: str,
    endpoint: str,
    preset: str,
    api_key: str,
    attempts: int = MAX_TRANSIENT_ATTEMPTS,
    outage_wait_seconds: int = MAX_OUTAGE_WAIT_SECONDS,
) -> tuple[str, str]:
    outage_started: float | None = None
    while True:
        response: requests.Response | None = None
        retry_error: ParaphraseAPIError | None = None
        for attempt in range(attempts):
            try:
                global _LAST_REQUEST_AT
                with _REQUEST_GATE:
                    now = time.monotonic()
                    wait = MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
                    if wait > 0:
                        time.sleep(wait)
                    _LAST_REQUEST_AT = time.monotonic()
                response = requests.post(
                    endpoint,
                    headers={"Authorization": api_key, "Content-Type": "application/json"},
                    json={"text": text, "preset": preset},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code in TRANSIENT_STATUS_CODES:
                    print(
                        f"[Paraphrase] transient HTTP {response.status_code} "
                        f"attempt={attempt + 1}/{attempts} chars={len(text)}"
                    )
                    if attempt + 1 < attempts:
                        retry_after = getattr(response, "headers", {}).get("Retry-After", "")
                        try:
                            delay = min(float(retry_after), 30.0) if retry_after else min(2 ** attempt, 30)
                        except ValueError:
                            delay = min(2 ** attempt, 30)
                        time.sleep(delay)
                        continue
                    retry_error = ParaphraseAPIError(
                        f"改写接口暂时不可用（HTTP {response.status_code}，已重试 {attempts} 次，"
                        f"请求字符数 {len(text)}，响应指纹 "
                        f"{hashlib.sha256(response.content).hexdigest()[:12]}）",
                        response.status_code,
                    )
                    break
                response.raise_for_status()
                break
            except ParaphraseAPIError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt + 1 < attempts:
                    time.sleep(2 ** attempt)
                    continue
                retry_error = ParaphraseAPIError(
                    f"改写接口网络异常（已重试 {attempts} 次）"
                )
                retry_error.__cause__ = exc
                break
            except requests.HTTPError as exc:
                status = response.status_code if response is not None else None
                raise ParaphraseAPIError(f"改写接口请求失败（HTTP {status or '未知'}）", status) from exc

        if retry_error is None:
            break
        if outage_wait_seconds <= 0:
            raise retry_error
        if outage_started is None:
            outage_started = time.monotonic()
            print("[Paraphrase] provider outage; pausing before retrying the same segment")
        elapsed = time.monotonic() - outage_started
        remaining = outage_wait_seconds - elapsed
        if remaining <= 0:
            raise ParaphraseAPIError(
                f"{retry_error}；上游持续不可用，已等待 {int(elapsed)} 秒"
            ) from retry_error
        time.sleep(min(OUTAGE_POLL_SECONDS, remaining))

    if response is None:
        raise ParaphraseAPIError("改写接口没有返回响应")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ParaphraseAPIError("改写接口返回了无效 JSON") from exc
    if payload.get("success") is False:
        raise ParaphraseAPIError("改写接口返回失败")
    rewritten = payload.get("paraphrased_text")
    if not isinstance(rewritten, str):
        data = payload.get("data")
        if isinstance(data, dict):
            rewritten = data.get("paraphrased_text") or data.get("text")
    if not isinstance(rewritten, str):
        raise ParaphraseAPIError("API 响应缺少 paraphrased_text")
    request_id = payload.get("request_id")
    return rewritten, request_id if isinstance(request_id, str) else ""


def _extract_marked_text_partial(
    response: str, expected: list[int]
) -> tuple[dict[int, str], list[int]]:
    """Return every intact segment without discarding partial batch success."""
    found: dict[int, str] = {}
    for match in _MARKER.finditer(response):
        index = int(match.group(1))
        if index in expected and index not in found:
            value = match.group(2).strip()
            # A provider can nest or move another segment marker inside this
            # boundary. Treat that member as missing; otherwise an internal
            # CASEG token can leak into the published article.
            if value and not _ANY_MARKER_TOKEN.search(value):
                found[index] = re.sub(r"\s+", " ", value)
    missing = [index for index in expected if index not in found]
    return found, missing


def _extract_marked_text(response: str, expected: list[int]) -> dict[int, str]:
    found, missing = _extract_marked_text_partial(response, expected)
    if missing:
        raise RuntimeError(f"改写响应缺少 {len(missing)} 个文本片段标记")
    return found


def _normalize_segment(value: str, original: str) -> str:
    value = value.strip()
    if not value:
        raise RuntimeError("改写接口返回了空文本片段")
    if _ANY_MARKER_TOKEN.search(value):
        # Never insert transport markers into article prose. The caller has
        # already used the required POST; retaining this one original segment
        # is safer than publishing a corrupted response.
        return original
    expected_tokens = _PROTECTED_TOKEN.findall(original)
    missing_tokens = [token for token in expected_tokens if token not in value]
    if missing_tokens:
        raise RuntimeError("改写接口返回时丢失了受保护内容占位符")
    if "\n" not in original:
        return re.sub(r"\s*\r?\n\s*", " ", value)
    return value


def _rewrite_segments(
    texts: list[str], endpoint: str, preset: str, api_key: str
) -> tuple[list[str], int, list[str], int]:
    rewritten = list(texts)
    batch: list[tuple[int, str]] = []
    batch_length = 0
    request_count = 0
    request_ids: list[str] = []
    fallback_segments = 0

    def rewrite_tokenless(segment: str, attempts: int = SEGMENT_ATTEMPTS) -> str:
        """Rewrite visible prose in 8-span batches; structure stays local."""
        nonlocal request_count, fallback_segments
        pieces = _PROTECTED_TOKEN.split(segment)
        tokens = _PROTECTED_TOKEN.findall(segment)
        rewritten_pieces = list(pieces)
        visible = [(index, piece) for index, piece in enumerate(pieces) if _has_visible_text(piece)]

        def normalize_visible(response: str, original: str) -> str:
            response = response.strip()
            if not response:
                raise RuntimeError("改写接口返回了空文本片段")
            if _ANY_MARKER_TOKEN.search(response):
                return original
            return re.sub(r"\s*\r?\n\s*", " ", response) if "\n" not in original else response

        def transient(exc: ParaphraseAPIError) -> bool:
            return exc.status_code is None or exc.status_code in TRANSIENT_STATUS_CODES

        def rewrite_single(piece: str) -> str:
            """Recover one failed span without ever sending article placeholders."""
            nonlocal request_count
            last_error: Exception | None = None
            delays = (0, *RECOVERY_DELAYS_SECONDS)
            for recovery_index, delay in enumerate(delays):
                if delay:
                    time.sleep(delay)
                try:
                    # Keep every short target inside a normal marked request
                    # with a disposable context span, then extract only target 0.
                    if len(piece.strip()) <= SHORT_SEGMENT_MAX_CHARS:
                        payload = (
                            f"CASEG000000START {piece} CASEG000000END\n"
                            "CASEG000001START 请保持原意并自然改写这段文字 CASEG000001END"
                        )
                        request_count += 1
                        response, request_id = _request(
                            payload, endpoint, preset, api_key, SHORT_SEGMENT_ATTEMPTS, 0
                        )
                        if request_id:
                            request_ids.append(request_id)
                        extracted, missing = _extract_marked_text_partial(response, [0])
                        if missing:
                            # The POST succeeded, but using an unframed reply
                            # would risk mixing disposable context into the
                            # article. Preserve this tiny original span; other
                            # spans in the semantic unit still use the API.
                            return piece
                        value = extracted[0]
                    else:
                        request_count += 1
                        response, request_id = _request(
                            piece, endpoint, preset, api_key, attempts, 0
                        )
                        if request_id:
                            request_ids.append(request_id)
                        value = response
                    return normalize_visible(value, piece)
                except ParaphraseAPIError as exc:
                    last_error = exc
                    if not transient(exc):
                        raise
                except RuntimeError as exc:
                    last_error = exc
                    # A short wrapper can occasionally lose CASEG markers.
                    # Retry the same bounded recovery shape after cooldown.
                if recovery_index == len(delays) - 1:
                    break
            if last_error is not None:
                raise last_error
            raise RuntimeError("改写接口没有返回可用文本")

        for offset in range(0, len(visible), MAX_BATCH_SEGMENTS):
            current = visible[offset : offset + MAX_BATCH_SEGMENTS]
            payload = "\n".join(
                f"CASEG{local:06d}START {piece} CASEG{local:06d}END"
                for local, (_piece_index, piece) in enumerate(current)
            )
            try:
                request_count += 1
                response, request_id = _request(payload, endpoint, preset, api_key, 3, 0)
                if request_id:
                    request_ids.append(request_id)
                expected = list(range(len(current)))
                extracted, missing = _extract_marked_text_partial(response, expected)
                for local, (piece_index, original) in enumerate(current):
                    if local in extracted:
                        rewritten_pieces[piece_index] = normalize_visible(
                            extracted[local], original
                        )
                if missing:
                    fallback_segments += len(missing)
                    for local in missing:
                        piece_index, original = current[local]
                        rewritten_pieces[piece_index] = rewrite_single(original)
            except RuntimeError as exc:
                recoverable = (
                    "文本片段标记" in str(exc)
                    or isinstance(exc, ParaphraseAPIError)
                    and transient(exc)
                )
                if not recoverable:
                    raise
                fallback_segments += len(current)
                for piece_index, original in current:
                    rewritten_pieces[piece_index] = rewrite_single(original)

        rebuilt: list[str] = []
        for index, piece in enumerate(rewritten_pieces):
            rebuilt.append(piece)
            if index < len(tokens):
                rebuilt.append(tokens[index])
        return "".join(rebuilt)

    def rewrite_tokenless_with_recovery(segment: str) -> str:
        """Bounded recovery for a protected segment, still using the same POST."""
        last_error: ParaphraseAPIError | None = None
        for attempt in range(2):
            try:
                return rewrite_tokenless(segment, SEGMENT_ATTEMPTS)
            except ParaphraseAPIError as exc:
                last_error = exc
                if exc.status_code not in TRANSIENT_STATUS_CODES and exc.status_code is not None:
                    raise
                if attempt == 0:
                    time.sleep(2)
        if last_error is not None:
            raise last_error
        raise RuntimeError("改写接口没有返回可用文本")

    def rewrite_batch(current: list[tuple[int, str]]) -> None:
        nonlocal request_count, fallback_segments
        # Rich segments contain local structure placeholders. Do not send
        # them to the provider at all: rewrite only visible spans and restore
        # every protected token locally. This removes the provider-dependent
        # placeholder-preservation failure from the normal path.
        protected_current = [item for item in current if _PROTECTED_TOKEN.search(item[1])]
        if protected_current:
            for index, value in protected_current:
                rewritten[index] = rewrite_tokenless_with_recovery(value)
            current = [item for item in current if not _PROTECTED_TOKEN.search(item[1])]
            if not current:
                return
        payload = "\n".join(
            f"CASEG{index:06d}START {value} CASEG{index:06d}END"
            for index, value in current
        )
        try:
            request_count += 1
            # Keep the successful small-batch request shape. If the provider
            # rejects a marked batch, recover immediately with plain-text
            # requests instead of recursively resubmitting marked payloads.
            response, request_id = _request(
                payload, endpoint, preset, api_key, 3, 0
            )
            if request_id:
                request_ids.append(request_id)
            extracted, missing = _extract_marked_text_partial(
                response, [index for index, _ in current]
            )
            for index, value in extracted.items():
                rewritten[index] = _normalize_segment(value, texts[index])
            if missing:
                # Retain successful members of this batch and recover only
                # the segments whose markers were actually removed.
                current = [item for item in current if item[0] in missing]
                raise RuntimeError(
                    f"改写响应缺少 {len(missing)} 个文本片段标记"
                )
        except RuntimeError as exc:
            is_marker_loss = (
                "文本片段标记" in str(exc)
                or "受保护内容占位符" in str(exc)
            )
            is_transient_batch_error = isinstance(exc, ParaphraseAPIError) and (
                exc.status_code is None or exc.status_code in TRANSIENT_STATUS_CODES
            )
            if not (is_marker_loss or is_transient_batch_error):
                raise
            # Marker loss or a persistent batch-level 5xx can be payload
            # specific. Re-send only this batch one segment at a time.
            fallback_segments += len(current)

            def is_transient(error: ParaphraseAPIError) -> bool:
                return error.status_code is None or error.status_code in TRANSIENT_STATUS_CODES

            def rewrite_plain(value: str, attempts: int) -> str:
                """Rewrite one segment, using only the configured POST."""
                nonlocal request_count

                def rewrite_one_visible_piece(piece: str) -> str:
                    """Rewrite plain visible text; never require structure tokens."""
                    nonlocal request_count
                    piece = piece.strip()
                    if not piece or not _has_visible_text(piece):
                        return piece

                    request_count += 1
                    if len(piece) <= SHORT_SEGMENT_MAX_CHARS:
                        # Never send a short span as a raw request. The
                        # provider has returned HTTP 500 for both tiny labels
                        # and 20-30 character headings. The same target in a
                        # two-span marked request is stable. The second span
                        # is transport-only context and is discarded locally.
                        candidate = (
                            f"CASEG000000START {piece} CASEG000000END\n"
                            "CASEG000001START 请保持原意并自然改写这段文字 CASEG000001END"
                        )
                        response, request_id = _request(
                            candidate,
                            endpoint,
                            preset,
                            api_key,
                            attempts=SHORT_SEGMENT_ATTEMPTS,
                            outage_wait_seconds=0,
                        )
                        if request_id:
                            request_ids.append(request_id)
                        extracted, missing = _extract_marked_text_partial(response, [0])
                        if missing:
                            # The target was sent through the required POST,
                            # but its boundary was removed. Keep the original
                            # short span rather than inserting ambiguous text
                            # or failing the complete article.
                            return piece
                        return extracted[0]

                    response, request_id = _request(
                        piece,
                        endpoint,
                        preset,
                        api_key,
                        attempts=attempts,
                        outage_wait_seconds=0,
                    )
                    if request_id:
                        request_ids.append(request_id)
                    response = response.strip()
                    if not response:
                        raise RuntimeError("改写接口返回了空文本片段")
                    return (
                        re.sub(r"\s*\r?\n\s*", " ", response)
                        if "\n" not in piece
                        else response
                    )

                def call(
                    candidate: str,
                    candidate_attempts: int,
                    replacements: dict[str, str] | None = None,
                ) -> str:
                    nonlocal request_count
                    request_count += 1
                    response, request_id = _request(
                        candidate,
                        endpoint,
                        preset,
                        api_key,
                        attempts=candidate_attempts,
                        outage_wait_seconds=0,
                    )
                    if request_id:
                        request_ids.append(request_id)
                    # Convert a provider-friendly placeholder spelling back
                    # to the canonical spelling before local validation.
                    if replacements:
                        for sent, canonical in replacements.items():
                            response = response.replace(sent, canonical)
                    return _normalize_segment(response, value)

                def rewrite_visible_spans(segment: str) -> str:
                    """Rewrite visible spans separately when markers are unstable."""
                    pieces = _PROTECTED_TOKEN.split(segment)
                    tokens = _PROTECTED_TOKEN.findall(segment)
                    rewritten_pieces: list[str] = []
                    for piece in pieces:
                        if not _has_visible_text(piece):
                            rewritten_pieces.append(piece)
                            continue
                        rewritten_pieces.append(rewrite_one_visible_piece(piece))
                    rebuilt: list[str] = []
                    for index, piece in enumerate(rewritten_pieces):
                        rebuilt.append(piece)
                        if index < len(tokens):
                            rebuilt.append(tokens[index])
                    return "".join(rebuilt)

                # Never send a short recovery segment as raw text. The
                # provider's intermittent 500 is payload-specific; the marked
                # target plus disposable context is stable.
                if not _PROTECTED_TOKEN.search(value) and len(value.strip()) <= SHORT_SEGMENT_MAX_CHARS:
                    return rewrite_one_visible_piece(value)

                try:
                    return call(value, attempts)
                except ParaphraseAPIError as exc:
                    # The marked request has already failed validation. Send
                    # only visible spans through the same POST and rebuild
                    # protected structure locally.
                    if _PROTECTED_TOKEN.search(value):
                        try:
                            return rewrite_visible_spans(value)
                        except (ParaphraseAPIError, RuntimeError):
                            pass
                    raise
                except RuntimeError as exc:
                    if "受保护内容占位符" not in str(exc):
                        raise
                    # Final recovery never sends structure tokens. Rewrite
                    # each visible span through the same POST and reinsert
                    # protected Markdown/HTML locally.
                    if _PROTECTED_TOKEN.search(value):
                        try:
                            return rewrite_visible_spans(value)
                        except (ParaphraseAPIError, RuntimeError):
                            pass
                    raise exc

            pending: list[tuple[int, str, ParaphraseAPIError]] = []
            for index, value in current:
                try:
                    rewritten[index] = rewrite_plain(value, SEGMENT_ATTEMPTS)
                except ParaphraseAPIError as exc:
                    if not is_transient(exc):
                        raise
                    pending.append((index, value, exc))

            # A provider 500 is often a short-lived upstream failure rather
            # than a bad segment. Retry only the failed segments after brief,
            # bounded cooldowns; never wait for the old 900-second circuit.
            for delay in RECOVERY_DELAYS_SECONDS:
                if not pending:
                    break
                time.sleep(delay)
                next_pending: list[tuple[int, str, ParaphraseAPIError]] = []
                for index, value, previous_error in pending:
                    try:
                        rewritten[index] = rewrite_plain(value, RECOVERY_ATTEMPTS)
                    except ParaphraseAPIError as exc:
                        if not is_transient(exc):
                            raise
                        next_pending.append((index, value, previous_error))
                pending = next_pending

            if pending:
                index, value, original_error = pending[0]
                segment_fingerprint = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
                raise ParaphraseAPIError(
                    f"{original_error}；失败片段索引 {index}，片段字符数 {len(value)}，"
                    f"片段指纹 {segment_fingerprint}；已完成 {len(current) - len(pending)}/{len(current)} 段恢复",
                    original_error.status_code,
                ) from original_error

    def flush() -> None:
        nonlocal batch, batch_length
        if batch:
            rewrite_batch(batch)
        batch = []
        batch_length = 0

    for index, value in enumerate(texts):
        item_length = len(value) + 40
        if batch and (
            len(batch) >= MAX_BATCH_SEGMENTS
            or batch_length + item_length > MAX_BATCH_CHARS
        ):
            flush()
        batch.append((index, value))
        batch_length += item_length
    flush()
    return rewritten, request_count, request_ids, fallback_segments


def _restore_segments(plan: _RewritePlan, texts: list[str]) -> str:
    result = plan.template
    for index, value in enumerate(texts):
        marker = re.compile(
            rf"CASEG{index:06d}START.*?CASEG{index:06d}END", re.DOTALL
        )
        result = marker.sub(lambda _: f"CASEG{index:06d}START{value}CASEG{index:06d}END", result, count=1)
    # Remove local markers after all replacements. The template still has one
    # marker pair per segment, so this is deterministic and safe.
    for index, value in reversed(list(enumerate(texts))):
        result = result.replace(f"CASEG{index:06d}START{plan.texts[index]}CASEG{index:06d}END", value)
        result = re.sub(
            rf"CASEG{index:06d}START(.*?)CASEG{index:06d}END",
            lambda match: match.group(1),
            result,
            flags=re.DOTALL,
        )
    for token, original in plan.protected.items():
        result = result.replace(token, original)
    return result


def _to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert(text)
    except ImportError:
        # The dependency is declared in pyproject.toml. Keep a deterministic
        # fallback for minimal deployments that have not synced dependencies.
        table = str.maketrans({"標": "标", "題": "题", "內": "内", "容": "容", "與": "与", "為": "为", "這": "这", "個": "个", "數": "数", "據": "据", "將": "将", "頁": "页", "圖": "图", "檔": "档", "說": "说", "產": "产", "業": "业", "後": "后", "發": "发", "現": "现", "間": "间", "時": "时", "處": "处", "長": "长", "開": "开", "關": "关", "門": "门", "實": "实", "驗": "验", "線": "线", "讀": "读", "寫": "写", "網": "网", "從": "从", "進": "进", "還": "还", "來": "来", "會": "会", "對": "对", "應": "应", "無": "无", "讓": "让", "種": "种", "問": "问", "題": "题", "專": "专", "業": "业", "與": "与"})
        return text.translate(table)


def _format_signature(text: str) -> tuple[list[str], list[str], list[tuple[str, str]], list[str], list[str], list[str]]:
    """Capture heading syntax and immutable structures for verification."""
    headings = re.findall(r"^\s{0,3}(#{1,6})\s+.*?$", text, re.MULTILINE)
    urls = re.findall(r"https?://[^\s<>\]\)]+", text)
    html_styles = []
    for tag in re.findall(r"</?[^>\r\n]+>", text):
        tag_name = (re.match(r"</?\s*([\w-]+)", tag) or ["", ""])[1].lower()
        style_match = re.search(r"\bstyle\s*=\s*(['\"])(.*?)\1", tag, re.IGNORECASE)
        html_styles.append((tag_name, style_match.group(2) if style_match else ""))
    inline_code = ["`" for _ in re.finditer(r"`[^`\r\n]*`", text)]
    fences = re.findall(r"^\s*(```|~~~).*?^\s*\1\s*$", text, re.MULTILINE | re.DOTALL)
    markdown_styles = re.findall(r"\*\*|__|~~", text)
    return headings, urls, html_styles, inline_code, fences, markdown_styles


def paraphraser_node(state: AgentState) -> dict:
    """Rewrite draft prose before any image/screenshot processing."""
    draft = state.get("draft", "")
    if not draft.strip():
        return {"draft": draft, "paraphrase_applied": False, "log": state.get("log", []) + ["改写跳过：文章为空"]}

    api_key = get_config("PARAPHRASE_API_KEY")
    endpoint = get_config("PARAPHRASE_API_URL", DEFAULT_ENDPOINT)
    preset = get_config("PARAPHRASE_PRESET", DEFAULT_PRESET)
    if not api_key:
        raise RuntimeError("未配置 PARAPHRASE_API_KEY，禁止跳过指定 POST 改写")

    plan = _build_plan(draft)
    if not plan.texts:
        return {"draft": draft, "paraphrase_applied": False, "log": state.get("log", []) + ["改写跳过：没有可改写的正文"]}

    print(f"\n[Paraphrase] 开始改写（{len(plan.texts)} 个文本片段，preset={preset}）...")
    try:
        # Serialize rewrite runs across simultaneous browser requests. The
        # provider is sensitive to bursty traffic, especially during fallback.
        with _PARAPHRASE_RUN_LOCK:
            rewritten_segments, request_count, request_ids, fallback_segments = _rewrite_segments(
                plan.texts, endpoint, preset, api_key
            )
        changed_count = sum(before != after for before, after in zip(plan.texts, rewritten_segments))
        if changed_count == 0:
            raise RuntimeError("指定 POST 返回内容与原文完全一致，没有产生实际改写")
        rewritten = _restore_segments(plan, rewritten_segments)
        rewritten = _to_simplified(rewritten)
        if _ANY_MARKER_TOKEN.search(rewritten) or _PROTECTED_TOKEN.search(rewritten):
            raise RuntimeError("改写结果仍含内部结构标记，禁止进入插图阶段")
        if _format_signature(draft) != _format_signature(rewritten):
            raise RuntimeError("标题或受保护排版结构校验不通过")
        heading_count = len(re.findall(r"^\s{0,3}#{1,6}\s+", draft, re.MULTILINE))
        request_proof = ", ".join(request_id[:8] for request_id in request_ids) or "服务未返回 request_id"
        print(f"  改写完成（接口调用 {request_count} 次，变化 {changed_count}/{len(plan.texts)} 段，标题格式已恢复并校验 {heading_count} 处）")
        return {
            "draft": rewritten,
            "paraphrase_applied": True,
            "paraphrase_request_count": request_count,
            "paraphrase_changed_count": changed_count,
            "paraphrase_request_ids": request_ids,
            "paraphrase_error": "",
            "log": state.get("log", []) + [
                f"✍️ 全文改写完成：指定 POST 调用 {request_count} 次，实际变化 {changed_count}/{len(plan.texts)} 段，请求凭证 {request_proof}，标题格式已恢复"
            ] + ([
                f"改写接口批次异常，已对其中 {fallback_segments} 段单独恢复；无法可靠识别边界的极短片段保持原文"
            ] if fallback_segments else []),
        }
    except Exception as exc:
        print(f"  [Paraphrase] 改写失败，终止生成：{type(exc).__name__}: {exc}")
        raise RuntimeError(f"指定 POST 改写失败，已停止进入插图阶段：{type(exc).__name__}: {exc}") from exc

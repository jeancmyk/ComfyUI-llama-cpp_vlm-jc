"""
Tool-call recovery for chat templates that leave the call in the text.

The problem
-----------
llama-cpp-python's vision chat handlers (``Qwen35ChatHandler`` and friends)
render tool definitions into the prompt using the Hermes/Qwen XML convention
and then never read the reply back. A model that follows those instructions
returns, as ordinary assistant **content**::

    <tool_call>
    <function=list_files>
    <parameter=path>
    C:\\Users\\Public\\Documents
    </parameter>
    </function>
    </tool_call>

with ``finish_reason="stop"``. An OpenAI client only acts on
``message.tool_calls``; with the call buried in the prose it sees a plain
answer, runs nothing, and the turn stalls. That is not a model bug - the model
did exactly what the template asked - so the endpoint has to parse it.

Two failures share this root cause, and both are handled here:

1. **Outbound** - every ``<tool_call>`` block is lifted out of the content into
   ``message.tool_calls`` (and ``finish_reason="tool_calls"``), for the blocking
   and the streaming reply alike. Surrounding prose is preserved as content.
2. **Inbound** - OpenAI carries ``function.arguments`` as a *JSON string*, but
   the Qwen template iterates them as a mapping (``arguments | items``), so
   replaying any tool result raises ``TypeError: Can only get item pairs from a
   mapping`` and fails the whole request. The Gemma4 template raises an explicit
   error for the same input. :func:`normalize_messages` deserializes the
   arguments once, before the template sees them.

Parameter values are re-typed from the tool's own JSON schema, because the XML
dialect carries no types: a ``timeoutMs`` declared ``integer`` arrives as the
string ``"600000"`` otherwise.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"

_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>")
# `<parameter=name>value</parameter>`, the dialect Qwen's template documents.
_PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</\s*parameter\s*>", re.S | re.I)
# `<parameter name="name">value</parameter>`, which several templates and models
# emit instead. Without this the block parses as a call with no arguments at all.
_PARAMETER_ATTR_RE = re.compile(
    r"<parameter\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</\s*parameter\s*>", re.S | re.I)
# An opening parameter tag in either spelling, used by the tolerant scanner
# below when a model forgets a closing tag or truncates the block.
_PARAM_OPEN_RE = re.compile(
    r"<parameter\s*(?:=\s*([^>\s]+)|name\s*=\s*[\"']([^\"']+)[\"'])\s*>", re.I)
_PARAM_CLOSE_RE = re.compile(r"</\s*parameter\s*>", re.I)
# Structural tags a truncated or sloppy block can leave at the end of a value.
_TRAILING_TAG_RE = re.compile(r"(?:\s*</?\s*(?:function|tool_call|arguments|parameter)\s*>)+$",
                              re.I)
# `<arguments>{...}</arguments>`, an inner wrapper around a JSON object.
_ARGUMENTS_RE = re.compile(r"<arguments\s*>(.*?)</arguments>", re.S)
# Leftover structural tags once the body of a call is isolated.
_TAG_RE = re.compile(r"</?(?:function|tool_call|arguments)\s*>", re.I)


# --------------------------------------------------------------------------- #
# schema helpers
# --------------------------------------------------------------------------- #
def _tool_index(tools: Optional[Iterable[Any]]) -> Dict[str, Dict[str, str]]:
    """``{tool name: {parameter name: declared JSON type}}``."""
    index: Dict[str, Dict[str, str]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = function.get("parameters")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        declared: Dict[str, str] = {}
        if isinstance(properties, dict):
            for key, spec in properties.items():
                if isinstance(spec, dict) and isinstance(spec.get("type"), str):
                    declared[key] = spec["type"]
        index[name] = declared
    return index


def _declared_type(index: Dict[str, Dict[str, str]], name: str, key: str) -> Optional[str]:
    return (index.get(name) or {}).get(key)


def _coerce(value: str, declared: Optional[str]) -> Any:
    """Re-type an XML parameter using the declared JSON schema type."""
    if not declared or declared == "string":
        return value
    if declared == "boolean":
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if declared == "object" and not isinstance(parsed, dict):
        return value
    if declared == "array" and not isinstance(parsed, list):
        return value
    if declared == "integer" and isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed


def _strip_edges(value: str) -> str:
    """Drop the newline the template wraps every parameter value in."""
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def arguments_map(value: Any) -> Dict[str, Any]:
    """Normalise anything callers hand us as ``function.arguments``."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_object(text: str) -> Optional[Dict]:
    """Parse ``text`` as a JSON object, tolerating stray XML leftovers.

    Models mix dialects freely: a ``<function=f>`` wrapper around a bare JSON
    body, or a fenced code block. The first balanced object is accepted rather
    than losing every argument to one stray character.
    """
    candidate = text.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except ValueError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start:end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _arguments_from_payload(payload: Dict) -> Tuple[Optional[str], Dict[str, Any]]:
    """Read ``(name, arguments)`` out of one parsed JSON object."""
    holder = payload.get("function")
    scoped = holder if isinstance(holder, dict) else payload
    candidate = scoped.get("name") or payload.get("name") or payload.get("tool")
    name = candidate if isinstance(candidate, str) and candidate else None
    for key in ("arguments", "parameters", "args", "input"):
        if key in scoped or key in payload:
            return name, arguments_map((scoped.get(key) if key in scoped else payload.get(key)))
    # Flat dialect: <tool_call>{"name": "f", "command": "..."}</tool_call>
    structural = {"name", "function", "tool", "type", "id"}
    return name, {k: v for k, v in scoped.items()
                  if k not in structural and not isinstance(v, dict)}


def _tolerant_arguments(body: str, index: Dict[str, Dict[str, str]],
                        name: Optional[str] = None) -> Dict[str, str]:
    """Collect raw parameter text when the closing tags are not trustworthy.

    A model may drop ``</parameter>``, misspell its case, or be cut off by the
    token budget mid-block. The strict patterns then match nothing and the call
    would reach the client with no arguments at all - so each value runs from
    its opening tag to the next opening tag (or the end), which recovers what
    the model did say without inventing anything.
    """
    opens = list(_PARAM_OPEN_RE.finditer(body))
    found: Dict[str, str] = {}
    for position, match in enumerate(opens):
        key = match.group(1) or match.group(2)
        if not key:
            continue
        end = opens[position + 1].start() if position + 1 < len(opens) \
            else len(body)
        raw = body[match.end():end]
        closing = _PARAM_CLOSE_RE.search(raw)
        if closing:
            raw = raw[:closing.start()]
        value = _TRAILING_TAG_RE.sub("", _strip_edges(raw)).strip()
        if value:
            found[key] = value
    return found


def _xml_arguments(name: str, body: str,
                   index: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Collect ``<parameter>`` values, in either attribute spelling.

    The tolerant scan runs first and the strict patterns fill any gap. It has to
    be this way round: when one parameter is closed properly and the next is not
    (a long ``content`` value whose ``</parameter>`` the model never got to, or
    a block cut short by the token budget), the strict patterns match the first
    parameter and the second is lost. The call then reaches the client with
    *some* of its arguments - ``write`` with ``file_path`` but no ``content`` -
    which is exactly the "missing required property" loop the model cannot see
    its way out of.
    """
    arguments: Dict[str, Any] = {}
    seen: set = set()
    for pattern in (_PARAMETER_RE, _PARAMETER_ATTR_RE):
        for parameter in pattern.finditer(body):
            key = parameter.group(1)
            raw = _strip_edges(parameter.group(2))
            value = _coerce(raw, _declared_type(index, name, key))
            if key in seen:
                previous = arguments[key]
                arguments[key] = (previous if isinstance(previous, list) else [previous]) + [value]
            else:
                arguments[key] = value
                seen.add(key)

    # Fill in whatever the strict patterns could not bound.
    for key, raw in _tolerant_arguments(body, index, name).items():
        if key not in arguments:
            arguments[key] = _coerce(raw, _declared_type(index, name, key))
    return arguments


def _parse_block(inner: str, index: Dict[str, Dict[str, str]], position: int) -> Optional[Dict]:
    """Turn the inside of one ``<tool_call>`` block into an OpenAI tool call."""
    text = inner.strip()
    if not text:
        return None

    name: Optional[str] = None
    arguments: Dict[str, Any] = {}

    payload = _json_object(text)
    if payload is not None:
        # JSON dialect: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        name, arguments = _arguments_from_payload(payload)

    if name is None:
        match = _FUNCTION_RE.search(text)
        if match is None:
            return None
        name = match.group(1)
        body = text[match.end():]
        arguments = _xml_arguments(name, body, index)
        if not arguments:
            # <function=f>{"command": "..."}</function> and
            # <function=f><arguments>{...}</arguments></function>: the body is a
            # JSON object rather than XML parameters. Recovering it here is what
            # keeps a real call from reaching the client with no arguments.
            for container in (_ARGUMENTS_RE.search(body), None):
                inner_json = _json_object(container.group(1)) if container else None
                if inner_json is None:
                    inner_json = _json_object(_TAG_RE.sub("", body))
                if inner_json is not None:
                    _, recovered = _arguments_from_payload(inner_json)
                    if recovered:
                        arguments = recovered
                        break
        if not arguments and name not in index:
            # "The format is <tool_call><function=example>...</function>" is
            # prose about the format, not a call: an unknown name with nothing
            # to pass is indistinguishable from an illustration, while a real
            # zero-argument tool is always a name the request declared.
            return None

    return {
        "id": f"call_{uuid.uuid4().hex[:16]}_{position}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def split_content(text: Optional[str], tools: Optional[Iterable[Any]]) -> Tuple[Optional[str], List[Dict]]:
    """Lift every ``<tool_call>`` block out of a finished message.

    Returns ``(prose, calls)``. ``prose`` is the text with the blocks removed
    (``None`` when nothing else was said, matching the OpenAI convention), and
    ``calls`` is empty when the message held no call at all - in which case the
    original text is returned untouched. A request that declared no tools never
    gets its text rewritten, however much it looks like the format.

    A closing ``</tool_call>`` may be missing when the token budget runs out
    mid-call; the block then ends at the following ``<tool_call>`` or at the end
    of the text, so a truncated call is still delivered.
    """
    if not text or not tools or CALL_OPEN not in text:
        return text, []

    index = _tool_index(tools)
    prose: List[str] = []
    calls: List[Dict] = []
    cursor = 0

    while True:
        start = text.find(CALL_OPEN, cursor)
        if start < 0:
            prose.append(text[cursor:])
            break
        end = text.find(CALL_CLOSE, start)
        if end < 0:
            following = text.find(CALL_OPEN, start + len(CALL_OPEN))
            end = len(text) if following < 0 else following
            block = text[start + len(CALL_OPEN):end]
            call = _parse_block(block, index, len(calls))
            if call is None:
                prose.append(text[cursor:])
            else:
                prose.append(text[cursor:start])
                calls.append(call)
            if following < 0:
                break
            cursor = end
            continue
        prose.append(text[cursor:start])
        call = _parse_block(text[start + len(CALL_OPEN):end], index, len(calls))
        if call is None:
            # Something we do not understand: leave it exactly as the model said.
            prose.append(text[start:end + len(CALL_CLOSE)])
        else:
            calls.append(call)
        cursor = end + len(CALL_CLOSE)

    if not calls:
        return text, []
    return ("".join(prose).strip() or None), calls


def missing_required(calls: Iterable[Dict], tools: Optional[Iterable[Any]]) -> List[str]:
    """Names of calls that omit an argument their own schema requires.

    A call can still parse cleanly and carry nothing: the model leaves the
    parameters out, or writes them in a dialect not recovered above. The client
    then rejects the call (``missing required property "command"``) and the
    model, seeing only that error, retries with the same empty call while the
    real problem stays invisible in the server log. Reporting it here makes the
    broken link in the chain obvious.
    """
    index: Dict[str, List[str]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        schema = function.get("parameters")
        if isinstance(name, str) and isinstance(schema, dict):
            required = schema.get("required")
            if isinstance(required, list):
                index[name] = [key for key in required if isinstance(key, str)]
    problems: List[str] = []
    for call in calls or []:
        function = call.get("function") or {}
        name = function.get("name")
        required = index.get(name) or []
        given = arguments_map(function.get("arguments"))
        wanted = [key for key in required if key not in given]
        if wanted:
            problems.append(f"{name} (missing {', '.join(wanted)})")
    return problems


class ContentToolSplitter:
    """Route ``<tool_call>`` blocks out of a *streamed* content channel.

    Mirrors :func:`split_content` incrementally: text is released as soon as it
    cannot be the start of a call, and a block is only buffered while it is
    still open. A block the model never closes is recovered at ``flush`` - and
    handed back as plain text if it does not parse, so prose that merely
    mentions the tag is never swallowed.
    """

    def __init__(self, tools: Optional[Iterable[Any]] = None):
        self.index = _tool_index(tools)
        # With no declared tools nothing can be a call: pass text straight on.
        self.enabled = bool(tools)
        self.pending = ""
        self.in_call = False
        self.calls: List[Dict] = []
        self._hold = len(CALL_OPEN) - 1

    def feed(self, text: Optional[str]) -> List[Tuple[str, Any]]:
        if not text:
            return []
        if not self.enabled:
            return [("content", text)]
        self.pending += text
        return self._drain(final=False)

    def flush(self) -> List[Tuple[str, Any]]:
        if not self.enabled:
            return []
        return self._drain(final=True)

    # -- internals ---------------------------------------------------------- #
    def _emit_call(self, inner: str, out: List[Tuple[str, Any]]) -> bool:
        call = _parse_block(inner, self.index, len(self.calls))
        if call is None:
            return False
        out.append(("tool_call", (len(self.calls), call)))
        self.calls.append(call)
        return True

    def _drain(self, final: bool) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []

        while True:
            if self.in_call:
                end = self.pending.find(CALL_CLOSE)
                if end >= 0:
                    inner = self.pending[len(CALL_OPEN):end]
                    rest = self.pending[end + len(CALL_CLOSE):]
                    if self._emit_call(inner, out):
                        self.pending = rest
                        self.in_call = False
                        continue
                    # Not a call: release the block verbatim and keep scanning.
                    out.append(("content", self.pending[:end + len(CALL_CLOSE)]))
                    self.pending = rest
                    self.in_call = False
                    continue
                if not final:
                    return out
                if not self._emit_call(self.pending[len(CALL_OPEN):], out):
                    out.append(("content", self.pending))
                self.pending = ""
                self.in_call = False
                return out

            start = self.pending.find(CALL_OPEN)
            if start >= 0:
                lead = self.pending[:start].rstrip()
                if lead:
                    out.append(("content", lead))
                self.pending = self.pending[start:]
                self.in_call = True
                continue

            if final:
                if self.pending:
                    out.append(("content", self.pending))
                self.pending = ""
                return out

            # Release everything that cannot be the head of a call.
            if len(self.pending) > self._hold:
                out.append(("content", self.pending[:len(self.pending) - self._hold]))
                self.pending = self.pending[len(self.pending) - self._hold:]
            return out


# --------------------------------------------------------------------------- #
# inbound normalisation
# --------------------------------------------------------------------------- #
def _text_of(content: Any) -> Any:
    """Tool results arrive as a string, or as OpenAI content parts."""
    if not isinstance(content, list):
        return content
    pieces = []
    for part in content:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            pieces.append(part["text"])
    return "\n".join(pieces)


def normalize_messages(messages: Optional[Iterable[Any]]) -> List[Any]:
    """Make an OpenAI-style history renderable by the XML chat templates.

    ``function.arguments`` is deserialized from its JSON string, and a tool
    result delivered as content parts is flattened to text. Everything else is
    passed through untouched.
    """
    out: List[Any] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        copied = dict(message)
        calls = copied.get("tool_calls")
        if isinstance(calls, list) and calls:
            fixed = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else call
                fixed.append({
                    **call,
                    "type": call.get("type") or "function",
                    "function": {
                        "name": function.get("name") or "",
                        "arguments": arguments_map(function.get("arguments")),
                    },
                })
            copied["tool_calls"] = fixed
        if copied.get("role") == "tool":
            copied["content"] = _text_of(copied.get("content"))
        out.append(copied)
    return out

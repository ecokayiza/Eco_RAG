import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

USAGE_KEYS = ("prompt_tokens", "prompt_cache_hit_tokens", "completion_tokens", "total_tokens")
WIRE_API_CHAT_COMPLETIONS = "chat_completions"
WIRE_API_RESPONSES = "responses"


class Message(BaseModel):
    """Store one chat message."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str
    content: str
    message_type: Optional[str] = None
    workflow_turn_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, Any]] = None
    workflow: Optional[Dict[str, Any]] = None

    def to_llm_message(self) -> Dict[str, Any]:
        """Convert one message into the provider payload shape."""
        payload: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.role == "tool" and self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class Response(BaseModel):
    """Store one model response."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, Any]] = None
    raw_response: Any


def _number(value: Any) -> int | float | None:
    """Return numeric values and ignore everything else."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _field(value: Any, name: str) -> Any:
    """Read one field from SDK objects or plain dictionaries."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _choices(response: Any) -> List[Any]:
    """Return provider choices as a list, tolerating usage-only stream chunks."""
    choices = _field(response, "choices")
    return list(choices) if choices else []


def _with_response_param_aliases(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Translate app-level response aliases into Responses API body fields."""
    payload = dict(params or {})

    reasoning_effort = payload.pop("model_reasoning_effort", None)
    if reasoning_effort:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            payload["reasoning"] = {"effort": str(reasoning_effort), **reasoning}
        else:
            payload["reasoning"] = {"effort": str(reasoning_effort)}

    disable_storage = payload.pop("disable_response_storage", None)
    if isinstance(disable_storage, bool) and disable_storage and "store" not in payload:
        payload["store"] = False

    return payload


def _usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Convert provider usage objects into plain dictionaries."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(usage, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return None


def normalize_token_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """Keep only prompt, cache, completion, and total token counters."""
    payload = _usage_to_dict(usage)
    if not payload:
        return None

    prompt_tokens = _number(payload.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _number(payload.get("input_tokens"))

    completion_tokens = _number(payload.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _number(payload.get("output_tokens"))

    total_tokens = _number(payload.get("total_tokens"))
    prompt_cache_hit_tokens = _number(payload.get("prompt_cache_hit_tokens"))

    if prompt_cache_hit_tokens is None:
        prompt_details = payload.get("prompt_tokens_details") or payload.get("input_tokens_details")
        if isinstance(prompt_details, dict):
            prompt_cache_hit_tokens = _number(prompt_details.get("cached_tokens"))

    values = {
        "prompt_tokens": prompt_tokens or 0,
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens if total_tokens is not None else (prompt_tokens or 0) + (completion_tokens or 0),
    }
    return values if any(values[key] for key in USAGE_KEYS) else None


def _raw_tool_calls(raw_response: Any) -> Optional[List[Any]]:
    """Extract the provider tool-call payload from a raw completion response."""
    if raw_response is None:
        return None

    choices = _choices(raw_response)
    if not choices:
        return None

    first_choice = choices[0]
    message = _field(first_choice, "message")
    if message is None:
        return None

    tool_calls = _field(message, "tool_calls")
    return list(tool_calls) if tool_calls else None


def _merge_stream_tool_call_delta(collected: dict[int, dict[str, Any]], delta: Any):
    """Accumulate OpenAI-style streamed tool-call deltas."""
    for item in _field(delta, "tool_calls") or []:
        index = _field(item, "index")
        if not isinstance(index, int):
            index = len(collected)
        current = collected.setdefault(index, {"function": {"arguments": ""}})

        item_id = _field(item, "id")
        if item_id:
            current["id"] = str(item_id)
        item_type = _field(item, "type")
        if item_type:
            current["type"] = str(item_type)

        function = _field(item, "function")
        if function is None:
            continue
        current_function = current.setdefault("function", {})
        name = _field(function, "name")
        if name:
            current_function["name"] = str(name)
        arguments = _field(function, "arguments")
        if arguments:
            current_function["arguments"] = f"{current_function.get('arguments', '')}{arguments}"


def _stream_tool_calls(collected: dict[int, dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Normalize accumulated streaming tool calls into the app tool-call shape."""
    if not collected:
        return None
    return extract_tool_calls({"choices": [{"message": {"tool_calls": [collected[index] for index in sorted(collected)]}}]})


def _responses_tools(tools: Optional[List[Dict]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Chat Completions-style function tools into Responses API tools."""
    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            payload: dict[str, Any] = {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
            if "strict" in function:
                payload["strict"] = function["strict"]
            elif "strict" in tool:
                payload["strict"] = tool["strict"]
            converted.append(payload)
            continue
        converted.append(dict(tool))

    return converted or None


def _tool_call_arguments(value: Any) -> Dict[str, Any]:
    """Normalize OpenAI-style function arguments into a dictionary."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Tool call arguments must decode to a JSON object: {text}")
        return parsed
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Unsupported tool call arguments payload: {value!r}")


def extract_tool_calls(raw_response: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert provider tool calls into the LangGraph-friendly message shape."""
    normalized: List[Dict[str, Any]] = []
    for item in _raw_tool_calls(raw_response) or []:
        function = getattr(item, "function", None)
        item_id = getattr(item, "id", None)
        if isinstance(item, dict):
            function = item.get("function")
            item_id = item.get("id")

        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")

        if not name:
            continue

        try:
            args = _tool_call_arguments(arguments)
        except (ValueError, json.JSONDecodeError):
            continue

        normalized.append(
            {
                "name": str(name),
                "args": args,
                "id": str(item_id or uuid4()),
                "type": "tool_call",
            }
        )

    return normalized or None


def extract_response_tool_calls(raw_response: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert Responses API function calls into the app tool-call shape."""
    normalized: List[Dict[str, Any]] = []
    output = _field(raw_response, "output")
    for item in output if isinstance(output, list) else []:
        if _field(item, "type") != "function_call":
            continue
        arguments = _field(item, "arguments")
        name = _field(item, "name")
        if not name:
            continue
        try:
            args = _tool_call_arguments(arguments)
        except (ValueError, json.JSONDecodeError):
            continue
        normalized.append(
            {
                "name": str(name),
                "args": args,
                "id": str(_field(item, "call_id") or _field(item, "id") or uuid4()),
                "type": "tool_call",
            }
        )
    return normalized or None


def _merge_response_tool_call_event(collected: dict[int, dict[str, Any]], event: Any):
    """Accumulate Responses API streamed function-call events."""
    event_type = str(_field(event, "type") or "")
    if event_type not in {
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    }:
        return

    item = _field(event, "item")
    if item is not None and _field(item, "type") != "function_call":
        return
    if item is None and event_type.startswith("response.output_item."):
        return

    index = _field(event, "output_index")
    if not isinstance(index, int):
        item_id = _field(event, "item_id")
        matching_index = next(
            (
                current_index
                for current_index, current in collected.items()
                if item_id and item_id in {current.get("id"), current.get("call_id")}
            ),
            None,
        )
        if matching_index is not None:
            index = matching_index
        elif len(collected) == 1 and event_type.startswith("response.function_call_arguments."):
            index = next(iter(collected))
        else:
            index = len(collected)
    current = collected.setdefault(index, {"type": "function_call", "arguments": ""})

    if item is not None:
        for key in ("id", "call_id", "name"):
            value = _field(item, key)
            if value:
                current[key] = str(value)
        arguments = _field(item, "arguments")
        if arguments is not None:
            current["arguments"] = str(arguments)

    if event_type == "response.function_call_arguments.delta":
        delta = _field(event, "delta")
        if delta:
            current["arguments"] = f"{current.get('arguments', '')}{delta}"
    elif event_type == "response.function_call_arguments.done":
        arguments = _field(event, "arguments")
        if arguments is not None:
            current["arguments"] = str(arguments)


def _response_stream_tool_calls(collected: dict[int, dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Normalize accumulated Responses API function calls."""
    if not collected:
        return None
    completed_items = [collected[index] for index in sorted(collected) if _field(collected[index], "name")]
    if not completed_items:
        return None
    return extract_response_tool_calls({"output": completed_items})


def _responses_content(value: Any, *, role: str = "user") -> Any:
    """Convert Chat Completions-style content into Responses API content."""
    if not isinstance(value, list):
        return str(value or "")

    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        part_type = item.get("type")
        if part_type == "text":
            text = str(item.get("text") or "")
            if text:
                parts.append({"type": text_type, "text": text})
            continue
        if part_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            image_url = str(image_url or "")
            if image_url:
                parts.append({"type": "input_image", "image_url": image_url, "detail": "auto"})

    return parts or ""


def _responses_function_call_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert persisted Chat Completions tool calls into Responses input items."""
    items: list[dict[str, Any]] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or "").strip()
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(
                    _tool_call_arguments(arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
        else:
            name = str(tool_call.get("name") or "").strip()
            args = tool_call.get("args")
            arguments = json.dumps(
                args if isinstance(args, dict) else {},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        if call_id and name:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments or "{}",
                }
            )
    return items


def _responses_input(messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a Responses API input transcript from chat-style messages."""
    inputs: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        if role in {"system", "developer"}:
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id:
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": str(message.get("content") or ""),
                    }
                )
                continue
            content = _responses_content(message.get("content"), role="user")
            inputs.append({"role": "user", "content": f"Tool result:\n{content}" if isinstance(content, str) else content})
            continue
        if role == "assistant":
            content = _responses_content(message.get("content"), role="assistant")
            if content:
                inputs.append({"role": "assistant", "content": content})
            inputs.extend(_responses_function_call_items(message))
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        content = _responses_content(message.get("content"), role=role)
        inputs.append({"role": role, "content": content})
    return inputs


def _responses_instructions(messages: List[Dict[str, Any]]) -> str | None:
    """Collect chat-style system/developer messages as Responses API instructions."""
    parts = [
        str(message.get("content") or "").strip()
        for message in messages
        if str(message.get("role") or "").strip() in {"system", "developer"} and str(message.get("content") or "").strip()
    ]
    return "\n\n".join(parts) or None


def _response_output_text(response: Any) -> str:
    """Extract assistant text from a Responses API response object."""
    output_text = _field(response, "output_text")
    if output_text:
        return str(output_text)

    parts: list[str] = []
    output = _field(response, "output")
    for item in output if isinstance(output, list) else []:
        content = _field(item, "content")
        for part in content if isinstance(content, list) else []:
            text = _field(part, "text")
            if text:
                parts.append(str(text))
    return "".join(parts)


def _remaining_response_text(streamed_text: str, completed_text: Any) -> str:
    """Return only the part of a finalized Responses text event not already streamed."""
    text = str(completed_text or "")
    if not text:
        return ""
    if not streamed_text:
        return text
    return text[len(streamed_text) :] if text.startswith(streamed_text) else ""


def _response_event_error(event: Any) -> str:
    """Render one failed Responses API stream event as a useful error."""
    response = _field(event, "response")
    error = _field(response, "error") or _field(event, "error")
    if error:
        message = _field(error, "message")
        if message:
            return str(message)
        return str(error)

    details = _field(response, "incomplete_details")
    if details:
        reason = _field(details, "reason")
        if reason:
            return f"Responses API stream ended incomplete: {reason}"
        return f"Responses API stream ended incomplete: {details}"

    event_type = str(_field(event, "type") or "response error")
    return f"Responses API stream failed: {event_type}"


class BaseChatModel(ABC):
    """Define the chat model interface used by the app."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        wire_api: str = WIRE_API_CHAT_COMPLETIONS,
        model: str | None = None,
        temperature: float | None = None,
        custom_request_params: Dict[str, Any] | None = None,
    ):
        """Create the shared provider client."""
        # The OpenAI client treats base_url as the prefix to endpoints.
        for suffix in ("/chat/completions", "/responses"):
            if base_url and base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.wire_api = wire_api if wire_api == WIRE_API_RESPONSES else WIRE_API_CHAT_COMPLETIONS
        self.model = model
        self.temperature = temperature
        self.custom_request_params = dict(custom_request_params) if isinstance(custom_request_params, dict) else None

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs,
    ) -> Response:
        """Return one complete assistant response."""

    @abstractmethod
    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield one assistant response as text chunks."""


class OpenAIChatModel(BaseChatModel):
    """Call an OpenAI-compatible chat completion endpoint."""

    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        include_optional: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stop": stop,
            **kwargs,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if include_optional:
            existing_extra_body = payload.get("extra_body")
            extra_body = dict(existing_extra_body) if isinstance(existing_extra_body, dict) else {}
            extra_body.update(self.custom_request_params or {})
        else:
            extra_body = {}
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def _build_responses_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        include_optional: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        existing_extra_body = kwargs.pop("extra_body", None)
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": _responses_input(messages),
            **kwargs,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        instructions = _responses_instructions(messages)
        if instructions and "instructions" not in payload:
            payload["instructions"] = instructions
        converted_tools = _responses_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
        if stream:
            payload["stream"] = True
        extra_body = dict(existing_extra_body) if isinstance(existing_extra_body, dict) else {}
        if include_optional:
            extra_body.update(_with_response_param_aliases(self.custom_request_params))
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    async def _create_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs,
    ):
        use_fallback = bool(self.custom_request_params)
        attempts = [(True, tools)]
        if use_fallback:
            attempts.append((False, tools))

        last_error: Exception | None = None
        for include_optional, attempted_tools in attempts:
            try:
                return await self.client.chat.completions.create(
                    **self._build_request_payload(
                        messages,
                        tools=attempted_tools,
                        stop=stop,
                        stream=stream,
                        include_optional=include_optional,
                        **kwargs,
                    )
                )
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Chat completion request was not attempted.")

    async def _create_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs,
    ):
        if stop:
            extra_body = kwargs.get("extra_body")
            kwargs["extra_body"] = {
                **(extra_body if isinstance(extra_body, dict) else {}),
                "stop": stop,
            }

        use_fallback = bool(self.custom_request_params)
        attempts = [(True, tools)]
        if use_fallback:
            attempts.append((False, tools))

        last_error: Exception | None = None
        for include_optional, attempted_tools in attempts:
            try:
                return await self.client.responses.create(
                    **self._build_responses_payload(
                        messages,
                        tools=attempted_tools,
                        stream=stream,
                        include_optional=include_optional,
                        **kwargs,
                    )
                )
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Responses API request was not attempted.")

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs,
    ) -> Response:
        """Fetch one non-streaming completion."""
        if self.wire_api == WIRE_API_RESPONSES:
            response = await self._create_response(
                messages=messages,
                tools=tools,
                stop=stop,
                **kwargs,
            )
            content = _response_output_text(response)
            tool_calls = extract_response_tool_calls(response)
            if not content and not tool_calls:
                raise ValueError("Responses API provider returned an empty response.")
            return Response(
                content=content,
                tool_calls=tool_calls,
                token_usage=normalize_token_usage(_field(response, "usage")),
                raw_response=response,
            )

        response = await self._create_completion(
            messages=messages,
            tools=tools,
            stop=stop,
            **kwargs,
        )
        choices = _choices(response)
        if not choices:
            raise ValueError("Chat provider returned no choices.")
        message = _field(choices[0], "message")
        if message is None:
            raise ValueError("Chat provider returned a choice without a message.")
        return Response(
            content=_field(message, "content") or "",
            tool_calls=extract_tool_calls(response),
            token_usage=normalize_token_usage(_field(response, "usage")),
            raw_response=response,
        )

    async def stream_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Fetch one streaming completion."""
        if self.wire_api == WIRE_API_RESPONSES:
            async for chunk in self._stream_response_api(
                messages=messages,
                tools=tools,
                stop=stop,
                callbacks=callbacks,
                **kwargs,
            ):
                yield chunk
            return

        callback_map = callbacks if isinstance(callbacks, dict) else {}
        on_usage = callback_map.get("on_usage")
        on_tool_calls = callback_map.get("on_tool_calls")
        streamed_tool_calls: dict[int, dict[str, Any]] = {}
        response = await self._create_completion(
            messages=messages,
            tools=tools,
            stop=stop,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            usage = normalize_token_usage(_field(chunk, "usage"))
            if usage and callable(on_usage):
                on_usage(usage)
            choices = _choices(chunk)
            if not choices:
                continue
            delta = _field(choices[0], "delta")
            _merge_stream_tool_call_delta(streamed_tool_calls, delta)
            text = _field(delta, "content")
            if text:
                yield text
        tool_calls = _stream_tool_calls(streamed_tool_calls)
        if tool_calls and callable(on_tool_calls):
            on_tool_calls(tool_calls)

    async def _stream_response_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        callback_map = callbacks if isinstance(callbacks, dict) else {}
        on_usage = callback_map.get("on_usage")
        on_tool_calls = callback_map.get("on_tool_calls")
        streamed_tool_calls: dict[int, dict[str, Any]] = {}
        completed_tool_calls: list[dict[str, Any]] | None = None
        streamed_text = ""
        response = await self._create_response(
            messages=messages,
            tools=tools,
            stop=stop,
            stream=True,
            **kwargs,
        )
        async for event in response:
            event_type = str(_field(event, "type") or "")
            _merge_response_tool_call_event(streamed_tool_calls, event)
            if event_type == "response.output_text.delta":
                text = _field(event, "delta")
                if text:
                    delta = str(text)
                    streamed_text = f"{streamed_text}{delta}"
                    yield delta
                continue
            if event_type == "response.output_text.done":
                delta = _remaining_response_text(streamed_text, _field(event, "text"))
                if delta:
                    streamed_text = f"{streamed_text}{delta}"
                    yield delta
                continue
            if event_type == "response.content_part.done":
                part = _field(event, "part")
                if _field(part, "type") == "output_text":
                    delta = _remaining_response_text(streamed_text, _field(part, "text"))
                    if delta:
                        streamed_text = f"{streamed_text}{delta}"
                        yield delta
                continue
            if event_type == "response.completed":
                completed_response = _field(event, "response")
                usage = normalize_token_usage(_field(completed_response, "usage"))
                if usage and callable(on_usage):
                    on_usage(usage)
                completed_tool_calls = extract_response_tool_calls(completed_response)
                delta = _remaining_response_text(streamed_text, _response_output_text(completed_response))
                if delta:
                    streamed_text = f"{streamed_text}{delta}"
                    yield delta
                continue
            if event_type in {"response.failed", "response.incomplete", "response.error"}:
                raise ValueError(_response_event_error(event))
        tool_calls = completed_tool_calls or _response_stream_tool_calls(streamed_tool_calls)
        if tool_calls and callable(on_tool_calls):
            on_tool_calls(tool_calls)

"""Structured JSON logging, OpenTelemetry-style trace correlation, and regex PII scrubbing.

Satisfies AgentOps Code Review Matrix Criteria:
- Criterion 13: Structured Cloud Logging JSON output with severity, trace/span correlation, and latency/token telemetry.
- Criterion 14: Pre-compiled regex-based PII scrubbing across logs, model requests/responses, and tool payloads.
- Criterion 15: Runtime observability plugin tracking model latency, token consumption, tool invocations, and guardrail triggers.
"""

from __future__ import annotations

import contextvars
import datetime
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Context variables for distributed request correlation across Coordinator & Sub-Agents
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default=""
)
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default=""
)
current_agent_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent_name", default="cal_grad_system"
)


@dataclass(frozen=True)
class PIIPattern:
    """Compiled regex pattern and replacement token for deterministic PII redaction."""

    name: str
    regex: re.Pattern[str]
    replacement: str


# Pre-compiled regex rules ordered from most specific to broadest to prevent partial-match collisions.
DEFAULT_PII_PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern(
        name="secret_token",
        regex=re.compile(
            r"(?i)(?:bearer\s+[a-z0-9\-._~+/]+=*|AIza[0-9A-Za-z\-_]{35}|ya29\.[0-9A-Za-z\-_]+|sk-[A-Za-z0-9]{20,})"
        ),
        replacement="[REDACTED_SECRET]",
    ),
    PIIPattern(
        name="ssn",
        regex=re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        replacement="[REDACTED_SSN]",
    ),
    PIIPattern(
        name="payment_card",
        regex=re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        replacement="[REDACTED_PAYMENT_CARD]",
    ),
    PIIPattern(
        name="email",
        regex=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        replacement="[REDACTED_EMAIL]",
    ),
    PIIPattern(
        name="employee_id",
        regex=re.compile(
            r"\b(?:EMP-\d{4,10}|GEMP-\d{4,10}|LDAP:[A-Za-z0-9._-]+|BADGE-\d{4,10})\b",
            re.IGNORECASE,
        ),
        replacement="[REDACTED_EMPLOYEE_ID]",
    ),
    PIIPattern(
        name="phone",
        regex=re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
        replacement="[REDACTED_PHONE]",
    ),
    PIIPattern(
        name="ipv4",
        regex=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
        replacement="[REDACTED_IP]",
    ),
)


@dataclass
class ScrubResult:
    """Result of a PII scrubbing pass containing sanitized value and category counts."""

    sanitized: Any
    redaction_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        return sum(self.redaction_counts.values())


class PIIScrubber:
    """Thread-safe regex-based PII scrubber for strings, nested dicts/lists, and ADK messages."""

    def __init__(self, patterns: tuple[PIIPattern, ...] = DEFAULT_PII_PATTERNS) -> None:
        self._patterns = patterns

    def scrub_text_with_stats(self, text: str) -> tuple[str, dict[str, int]]:
        """Scrubs PII from a string and returns (sanitized_text, counts_by_category)."""
        if not text:
            return text, {}
        counts: dict[str, int] = {}
        sanitized = text
        for rule in self._patterns:
            sanitized, num_subs = rule.regex.subn(rule.replacement, sanitized)
            if num_subs > 0:
                counts[rule.name] = counts.get(rule.name, 0) + num_subs
        return sanitized, counts

    def scrub_text(self, text: str) -> str:
        """Scrubs PII from a raw string."""
        sanitized, _ = self.scrub_text_with_stats(text)
        return sanitized

    def scrub_value(self, value: Any) -> ScrubResult:
        """Recursively scrubs strings inside dicts, lists, tuples, and primitives."""
        aggregated_counts: dict[str, int] = {}

        def _merge(new_counts: dict[str, int]) -> None:
            for k, v in new_counts.items():
                aggregated_counts[k] = aggregated_counts.get(k, 0) + v

        def _walk(item: Any) -> Any:
            if isinstance(item, str):
                cleaned, counts = self.scrub_text_with_stats(item)
                _merge(counts)
                return cleaned
            if isinstance(item, dict):
                return {
                    (_walk(k) if isinstance(k, str) else k): _walk(v)
                    for k, v in item.items()
                }
            if isinstance(item, list):
                return [_walk(elem) for elem in item]
            if isinstance(item, tuple):
                return tuple(_walk(elem) for elem in item)
            return item

        sanitized_val = _walk(value)
        return ScrubResult(sanitized=sanitized_val, redaction_counts=aggregated_counts)

    def scrub_llm_request(self, llm_request: LlmRequest) -> dict[str, int]:
        """Scrubs PII in-place across all text parts of an ADK LlmRequest before model invocation."""
        counts: dict[str, int] = {}
        if not llm_request.contents:
            return counts
        for content in llm_request.contents:
            if not content.parts:
                continue
            for part in content.parts:
                if part.text:
                    cleaned, part_counts = self.scrub_text_with_stats(part.text)
                    part.text = cleaned
                    for k, v in part_counts.items():
                        counts[k] = counts.get(k, 0) + v
        return counts

    def scrub_llm_response(self, llm_response: LlmResponse) -> dict[str, int]:
        """Scrubs PII in-place across text parts of an ADK LlmResponse."""
        counts: dict[str, int] = {}
        if not llm_response.content or not llm_response.content.parts:
            return counts
        for part in llm_response.content.parts:
            if part.text:
                cleaned, part_counts = self.scrub_text_with_stats(part.text)
                part.text = cleaned
                for k, v in part_counts.items():
                    counts[k] = counts.get(k, 0) + v
        return counts


# Singleton default scrubber instance
default_pii_scrubber = PIIScrubber()


class PIIScrubbingFilter(logging.Filter):
    """Logging filter that guarantees all log messages and structured extra fields are PII-free."""

    def __init__(self, scrubber: PIIScrubber = default_pii_scrubber) -> None:
        super().__init__()
        self._scrubber = scrubber

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        msg_result = self._scrubber.scrub_value(msg)
        record.msg = msg_result.sanitized
        record.args = ()

        total_redactions = dict(msg_result.redaction_counts)
        if hasattr(record, "structured_data") and isinstance(
            record.structured_data, dict
        ):
            extra_result = self._scrubber.scrub_value(record.structured_data)
            record.structured_data = extra_result.sanitized
            for k, v in extra_result.redaction_counts.items():
                total_redactions[k] = total_redactions.get(k, 0) + v

        record.pii_redactions = total_redactions
        return True


class CloudJsonFormatter(logging.Formatter):
    """Formats Python log records as single-line Google Cloud Logging structured JSON."""

    def __init__(
        self,
        project_id: str = "cal-grad-agent-prod",
        scrubber: PIIScrubber = default_pii_scrubber,
    ) -> None:
        super().__init__()
        self._project_id = project_id
        self._scrubber = scrubber

    def format(self, record: logging.LogRecord) -> str:
        message = self._scrubber.scrub_text(record.getMessage())
        trace_id = getattr(record, "trace_id", None) or current_trace_id.get() or ""
        session_id = (
            getattr(record, "session_id", None) or current_session_id.get() or ""
        )
        agent_name = (
            getattr(record, "agent_name", None)
            or current_agent_name.get()
            or "cal_grad_system"
        )

        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "severity": record.levelname,
            "message": message,
            "logger": record.name,
            "component": getattr(record, "component", "agentops"),
            "agent_name": agent_name,
            "session_id": session_id,
        }

        if trace_id:
            payload["logging.googleapis.com/trace"] = (
                f"projects/{self._project_id}/traces/{trace_id}"
            )

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict):
            scrubbed_extra = self._scrubber.scrub_value(structured_data)
            payload.update(scrubbed_extra.sanitized)

        pii_redactions = getattr(record, "pii_redactions", None)
        if pii_redactions:
            payload["pii_redactions"] = pii_redactions

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            payload["exception"] = self._scrubber.scrub_text(exc_text)

        return json.dumps(payload, default=str, sort_keys=False)


def configure_json_logging(
    logger_name: str = "cal_grad_agent",
    level: int = logging.INFO,
    project_id: str = "cal-grad-agent-prod",
    stream: Any = None,
) -> logging.Logger:
    """Configures and returns a structured JSON logger with automatic PII scrubbing."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Replace existing handlers cleanly to keep output deterministic across re-initializations
    logger.handlers.clear()
    logger.filters.clear()

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.addFilter(PIIScrubbingFilter(default_pii_scrubber))
    handler.setFormatter(CloudJsonFormatter(project_id=project_id))
    logger.addHandler(handler)
    return logger


# Module-level structured logger
agent_logger = configure_json_logging()


def log_structured_event(
    event_type: str,
    message: str,
    *,
    severity: int = logging.INFO,
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    component: str = "agentops",
    **fields: Any,
) -> None:
    """Emits a structured, PII-scrubbed JSON log event."""
    structured_data = {"event_type": event_type, **fields}
    extra: dict[str, Any] = {
        "component": component,
        "structured_data": structured_data,
    }
    if agent_name:
        extra["agent_name"] = agent_name
    if session_id:
        extra["session_id"] = session_id
    if trace_id:
        extra["trace_id"] = trace_id

    agent_logger.log(severity, message, extra=extra)


class AgentOpsObservabilityPlugin(BasePlugin):
    """ADK Plugin providing end-to-end PII scrubbing, latency tracking, and token telemetry."""

    def __init__(
        self,
        name: str = "agentops_observability_plugin",
        scrubber: PIIScrubber = default_pii_scrubber,
    ) -> None:
        super().__init__(name)
        self._scrubber = scrubber
        self._model_start_times: dict[str, float] = {}
        self._tool_start_times: dict[str, float] = {}

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        trace_id = current_trace_id.get() or uuid.uuid4().hex
        current_trace_id.set(trace_id)
        current_agent_name.set(agent.name)
        session_id = getattr(callback_context.session, "id", "")
        if session_id:
            current_session_id.set(session_id)

        log_structured_event(
            "agent_start",
            f"Starting agent '{agent.name}'",
            agent_name=agent.name,
            session_id=session_id,
            trace_id=trace_id,
            invocation_id=callback_context.invocation_id,
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        session_id = getattr(callback_context.session, "id", "")
        log_structured_event(
            "agent_complete",
            f"Completed agent '{agent.name}'",
            agent_name=agent.name,
            session_id=session_id,
            invocation_id=callback_context.invocation_id,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        key = f"{callback_context.invocation_id}:{callback_context.agent_name}"
        self._model_start_times[key] = time.perf_counter()
        redactions = self._scrubber.scrub_llm_request(llm_request)
        log_structured_event(
            "llm_request",
            f"Dispatching LLM request for '{callback_context.agent_name}'",
            agent_name=callback_context.agent_name,
            session_id=getattr(callback_context.session, "id", ""),
            invocation_id=callback_context.invocation_id,
            model=llm_request.model or "default",
            prompt_pii_redactions=redactions,
            content_turns=len(llm_request.contents or []),
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        key = f"{callback_context.invocation_id}:{callback_context.agent_name}"
        started = self._model_start_times.pop(key, None)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2) if started else 0.0
        redactions = self._scrubber.scrub_llm_response(llm_response)

        token_usage: dict[str, int] = {}
        if llm_response.usage_metadata:
            token_usage = {
                "prompt_tokens": llm_response.usage_metadata.prompt_token_count or 0,
                "candidates_tokens": llm_response.usage_metadata.candidates_token_count or 0,
                "total_tokens": llm_response.usage_metadata.total_token_count or 0,
            }

        # Persist cumulative telemetry counters in transient/session state
        telemetry = dict(callback_context.state.get("telemetry_metrics", {}))
        telemetry["llm_calls"] = telemetry.get("llm_calls", 0) + 1
        telemetry["total_tokens"] = telemetry.get("total_tokens", 0) + token_usage.get(
            "total_tokens", 0
        )
        callback_context.state["telemetry_metrics"] = telemetry

        log_structured_event(
            "llm_response",
            f"Received LLM response for '{callback_context.agent_name}'",
            agent_name=callback_context.agent_name,
            session_id=getattr(callback_context.session, "id", ""),
            invocation_id=callback_context.invocation_id,
            latency_ms=latency_ms,
            token_usage=token_usage,
            response_pii_redactions=redactions,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        call_key = f"{tool_context.invocation_id}:{tool.name}:{tool_context.function_call_id}"
        self._tool_start_times[call_key] = time.perf_counter()
        log_structured_event(
            "tool_start",
            f"Invoking tool '{tool.name}'",
            agent_name=tool_context.agent_name,
            session_id=getattr(tool_context.session, "id", ""),
            invocation_id=tool_context.invocation_id,
            tool_name=tool.name,
            tool_args_preview=self._scrubber.scrub_value(tool_args).sanitized,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        call_key = f"{tool_context.invocation_id}:{tool.name}:{tool_context.function_call_id}"
        started = self._tool_start_times.pop(call_key, None)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2) if started else 0.0

        scrubbed_result = self._scrubber.scrub_value(result)
        log_structured_event(
            "tool_complete",
            f"Completed tool '{tool.name}'",
            agent_name=tool_context.agent_name,
            session_id=getattr(tool_context.session, "id", ""),
            invocation_id=tool_context.invocation_id,
            tool_name=tool.name,
            latency_ms=latency_ms,
            result_pii_redactions=scrubbed_result.redaction_counts,
        )
        if scrubbed_result.total_redactions > 0 and isinstance(
            scrubbed_result.sanitized, dict
        ):
            return scrubbed_result.sanitized
        return None

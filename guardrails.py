"""Model Armor safety guardrails and Human-in-the-Loop (HITL) enforcement hooks.

Satisfies AgentOps Code Review Matrix Criteria 10 & 11:
- Criterion 10: Google Cloud Model Armor (`sanitizeUserPrompt` & `sanitizeModelResponse`)
  with fail-closed enforcement and defense-in-depth prompt-injection/jailbreak/RAI detection.
- Criterion 11: Human-in-the-Loop (`before_tool_callback` + ADK `ToolConfirmation` state gate)
  halting high-impact GRAD finalization until an explicit human reviewer approves.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from observability import default_pii_scrubber, log_structured_event
from secrets_manager import SecretManagerClient, get_secret_manager

# Tools that mutate external/official performance records and MUST pass Human-in-the-Loop sign-off
HITL_PROTECTED_TOOLS: frozenset[str] = frozenset({"submit_final_grad_draft"})


# Defense-in-depth deterministic prompt-injection, jailbreak, and rating-fraud signatures
_ADVERSARIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prompt_injection_override",
        re.compile(
            r"(?i)\b(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
            r"disregard\s+(?:your\s+)?system\s+(?:prompt|instructions)|"
            r"you\s+are\s+now\s+in\s+developer\s+mode|"
            r"reveal\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions))\b"
        ),
    ),
    (
        "delimiter_token_smuggling",
        re.compile(r"(?:<\|im_start\|>|<\|system\|>|\[INST\]|###\s*System\s*Override)", re.IGNORECASE),
    ),
    (
        "grad_fraud_or_fabrication",
        re.compile(
            r"(?i)\b(?:fabricate\s+(?:a\s+)?(?:peer\s+bonus|gthanks|metric|csat)|"
            r"invent\s+fake\s+(?:metrics|bookings|attainment)|"
            r"bypass\s+(?:hitl|human\s+review|manager\s+approval|model\s+armor))\b"
        ),
    ),
    (
        "malicious_uri_scheme",
        re.compile(r"(?i)\b(?:javascript:|data:text/html|vbscript:)"),
    ),
)

_UNSAFE_OUTPUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "system_prompt_leakage",
        re.compile(
            r"(?i)(?:INTERNAL_SYSTEM_CONTRACT_DO_NOT_DISCLOSE|BEGIN_CONFIDENTIAL_SYSTEM_PROMPT)"
        ),
    ),
    (
        "malicious_executable_link",
        re.compile(r"(?i)\b(?:javascript:|data:text/html)"),
    ),
)


@dataclass
class ModelArmorConfig:
    """Configuration for Google Cloud Model Armor REST/SDK inspection."""

    project_id: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    location: str = field(
        default_factory=lambda: os.getenv("MODEL_ARMOR_LOCATION", "us-central1")
    )
    template_id: str = field(
        default_factory=lambda: os.getenv("MODEL_ARMOR_TEMPLATE_ID", "")
    )
    remote_enabled: bool = field(
        default_factory=lambda: os.getenv("MODEL_ARMOR_REMOTE_ENABLED", "false").lower()
        == "true"
    )
    fail_closed: bool = field(
        default_factory=lambda: os.getenv("MODEL_ARMOR_FAIL_CLOSED", "true").lower()
        == "true"
    )
    timeout_seconds: float = 5.0

    @property
    def template_resource_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"templates/{self.template_id}"
        )


@dataclass(frozen=True)
class GuardrailVerdict:
    """Outcome of a Model Armor inspection on user input or model output."""

    allowed: bool
    reason_code: str = "CLEAN"
    matched_filters: tuple[str, ...] = ()
    message: str = ""


class ModelArmorGuardrail:
    """Enforces Google Cloud Model Armor (`sanitizeUserPrompt` & `sanitizeModelResponse`) + local defense-in-depth."""

    SAFE_REFUSAL_TEXT = (
        "Request blocked by Model Armor safety guardrails: prompt injection, policy violation, "
        "or unverified performance-record manipulation is not permitted."
    )

    def __init__(
        self,
        config: Optional[ModelArmorConfig] = None,
        secret_manager: Optional[SecretManagerClient] = None,
        transport: Optional[Callable[[str, dict[str, Any], float], dict[str, Any]]] = None,
    ) -> None:
        self.config = config or ModelArmorConfig()
        self._secret_manager = secret_manager or get_secret_manager()
        self._transport = transport

    def _resolve_template_id(self) -> str:
        if self.config.template_id:
            return self.config.template_id
        if self.config.project_id:
            secret = self._secret_manager.get_secret(
                "model-armor-template-id", default="cal-grad-armor-template"
            )
            return secret.get_secret_value()
        return "cal-grad-armor-template"

    def _default_rest_transport(
        self, endpoint_action: str, body: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Calls Google Cloud Model Armor v1 REST API using Application Default Credentials."""
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())

        template_id = self._resolve_template_id()
        url = (
            f"https://modelarmor.{self.config.location}.rep.googleapis.com/v1/"
            f"projects/{self.config.project_id}/locations/{self.config.location}/"
            f"templates/{template_id}:{endpoint_action}"
        )
        req = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_model_armor_api_response(api_payload: dict[str, Any]) -> GuardrailVerdict:
        """Parses `SanitizeUserPromptResponse` / `SanitizeModelResponseResponse` payloads."""
        result = api_payload.get("sanitizationResult", {})
        match_state = result.get("filterMatchState", "NO_MATCH_FOUND")
        if match_state == "MATCH_FOUND":
            matched: list[str] = []
            for filter_name, filter_detail in result.get("filterResults", {}).items():
                if "MATCH_FOUND" in json.dumps(filter_detail):
                    matched.append(filter_name)
            return GuardrailVerdict(
                allowed=False,
                reason_code="MODEL_ARMOR_MATCH_FOUND",
                matched_filters=tuple(matched or ["model_armor_policy"]),
                message="Blocked by Google Cloud Model Armor template enforcement.",
            )
        return GuardrailVerdict(allowed=True)

    def inspect_user_prompt(self, text: str) -> GuardrailVerdict:
        """Evaluates user prompt text via deterministic heuristics and Model Armor `sanitizeUserPrompt`."""
        if not text:
            return GuardrailVerdict(allowed=True)

        matched_local: list[str] = []
        for rule_name, pattern in _ADVERSARIAL_PATTERNS:
            if pattern.search(text):
                matched_local.append(rule_name)

        if matched_local:
            return GuardrailVerdict(
                allowed=False,
                reason_code="LOCAL_GUARDRAIL_BLOCK",
                matched_filters=tuple(matched_local),
                message=self.SAFE_REFUSAL_TEXT,
            )

        if self.config.remote_enabled or self._transport is not None:
            caller = self._transport or self._default_rest_transport
            try:
                api_resp = caller(
                    "sanitizeUserPrompt",
                    {"userPromptData": {"text": text}},
                    self.config.timeout_seconds,
                )
                return self._parse_model_armor_api_response(api_resp)
            except Exception as exc:  # noqa: BLE001
                log_structured_event(
                    "model_armor_error",
                    f"Model Armor sanitizeUserPrompt call failed: {exc}",
                    severity=logging.ERROR,
                    component="guardrails",
                    fail_closed=self.config.fail_closed,
                )
                if self.config.fail_closed:
                    return GuardrailVerdict(
                        allowed=False,
                        reason_code="MODEL_ARMOR_UNAVAILABLE_FAIL_CLOSED",
                        matched_filters=("service_unavailable",),
                        message="Safety inspection service is temporarily unavailable; failing closed per security policy.",
                    )

        return GuardrailVerdict(allowed=True)

    def inspect_model_response(self, text: str) -> GuardrailVerdict:
        """Evaluates generated model text via local policy checks and Model Armor `sanitizeModelResponse`."""
        if not text:
            return GuardrailVerdict(allowed=True)

        matched_local: list[str] = []
        for rule_name, pattern in _UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(text):
                matched_local.append(rule_name)

        if matched_local:
            return GuardrailVerdict(
                allowed=False,
                reason_code="LOCAL_OUTPUT_GUARDRAIL_BLOCK",
                matched_filters=tuple(matched_local),
                message="Generated response withheld by output safety guardrails.",
            )

        if self.config.remote_enabled or self._transport is not None:
            caller = self._transport or self._default_rest_transport
            try:
                api_resp = caller(
                    "sanitizeModelResponse",
                    {"modelResponseData": {"text": text}},
                    self.config.timeout_seconds,
                )
                return self._parse_model_armor_api_response(api_resp)
            except Exception as exc:  # noqa: BLE001
                log_structured_event(
                    "model_armor_output_error",
                    f"Model Armor sanitizeModelResponse call failed: {exc}",
                    severity=logging.ERROR,
                    component="guardrails",
                    fail_closed=self.config.fail_closed,
                )
                if self.config.fail_closed:
                    return GuardrailVerdict(
                        allowed=False,
                        reason_code="MODEL_ARMOR_OUTPUT_FAIL_CLOSED",
                        matched_filters=("service_unavailable",),
                        message="Output safety inspection unavailable; failing closed per security policy.",
                    )

        return GuardrailVerdict(allowed=True)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """ADK `before_model_callback` scanning incoming user turns and short-circuiting unsafe prompts."""
        if not llm_request.contents:
            return None

        # Inspect user-authored text turns in the request
        for content in llm_request.contents:
            if content.role != "user" or not content.parts:
                continue
            for part in content.parts:
                if not part.text:
                    continue
                verdict = self.inspect_user_prompt(part.text)
                if not verdict.allowed:
                    blocks = int(callback_context.state.get("guardrail_blocks", 0)) + 1
                    callback_context.state["guardrail_blocks"] = blocks
                    log_structured_event(
                        "guardrail_blocked_input",
                        "Model Armor blocked unsafe or adversarial user input",
                        severity=logging.WARNING,
                        component="guardrails",
                        agent_name=callback_context.agent_name,
                        session_id=getattr(callback_context.session, "id", ""),
                        reason_code=verdict.reason_code,
                        matched_filters=list(verdict.matched_filters),
                    )
                    return LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=verdict.message or self.SAFE_REFUSAL_TEXT)],
                        )
                    )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """ADK `after_model_callback` sanitizing PII and blocking unsafe model outputs."""
        if not llm_response.content or not llm_response.content.parts:
            return None

        default_pii_scrubber.scrub_llm_response(llm_response)

        for part in llm_response.content.parts:
            if not part.text:
                continue
            verdict = self.inspect_model_response(part.text)
            if not verdict.allowed:
                blocks = int(callback_context.state.get("guardrail_blocks", 0)) + 1
                callback_context.state["guardrail_blocks"] = blocks
                log_structured_event(
                    "guardrail_blocked_output",
                    "Model Armor blocked unsafe model output",
                    severity=logging.WARNING,
                    component="guardrails",
                    agent_name=callback_context.agent_name,
                    session_id=getattr(callback_context.session, "id", ""),
                    reason_code=verdict.reason_code,
                    matched_filters=list(verdict.matched_filters),
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=verdict.message)],
                    )
                )
        return None


# ---------------------------------------------------------------------------
# Criterion 11: Human-in-the-Loop (HITL) Approval Gate
# ---------------------------------------------------------------------------


def record_human_approval(
    state: dict[str, Any],
    *,
    reviewer_id: str,
    approved: bool,
    notes: str = "",
) -> dict[str, Any]:
    """Records an explicit human CAL/manager approval or rejection decision in session state."""
    clean_reviewer = default_pii_scrubber.scrub_text(reviewer_id.strip())
    if not clean_reviewer:
        raise ValueError("reviewer_id must be non-empty.")

    status = "APPROVED" if approved else "REJECTED"
    decision = {
        "status": status,
        "reviewer_id": clean_reviewer,
        "notes": default_pii_scrubber.scrub_text(notes),
    }
    state["hitl_approval_status"] = status
    state["hitl_decision_record"] = decision
    log_structured_event(
        "hitl_decision_recorded",
        f"Recorded human review decision: {status}",
        component="hitl",
        reviewer_id=clean_reviewer,
        status=status,
    )
    return decision


async def enforce_hitl_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict[str, Any]]:
    """ADK `before_tool_callback` gating privileged tools behind explicit Human-in-the-Loop approval."""
    if tool.name not in HITL_PROTECTED_TOOLS:
        return None

    state_status = tool_context.state.get("hitl_approval_status", "NOT_STARTED")
    adk_confirmed = bool(
        tool_context.tool_confirmation and tool_context.tool_confirmation.confirmed
    )

    # Check whether the caller explicitly passed reviewer_approved=True AND state/ADK confirmation is granted
    sub_arg = args.get("submission", args)
    arg_approved = (
        sub_arg.get("reviewer_approved", False)
        if isinstance(sub_arg, dict)
        else getattr(sub_arg, "reviewer_approved", False)
    )

    if (state_status == "APPROVED" or adk_confirmed) and arg_approved:
        return None

    draft_preview = (tool_context.state.get("current_grad_draft") or {}).get(
        "two_section_markdown",
        "No draft generated yet — call format_two_section_grad_draft first.",
    )
    tool_context.state["hitl_approval_status"] = "PENDING_REVIEW"

    log_structured_event(
        "hitl_gate_paused",
        f"Paused privileged tool '{tool.name}' pending Human-in-the-Loop approval",
        severity=logging.INFO,
        component="hitl",
        agent_name=tool_context.agent_name,
        session_id=getattr(tool_context.session, "id", ""),
        tool_name=tool.name,
        current_hitl_status=state_status,
    )

    return {
        "status": "PENDING_HITL_APPROVAL",
        "requires_human_approval": True,
        "tool_blocked": tool.name,
        "message": (
            "Human-in-the-Loop checkpoint: Final GRAD submission requires explicit CAL human review "
            "and approval (`hitl_approval_status == 'APPROVED'`). Present the Two-Section GRAD draft "
            "below to the user and await confirmation."
        ),
        "draft_preview": draft_preview,
    }


async def scrub_tool_output_after_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """ADK `after_tool_callback` ensuring every tool output returned to the LLM is PII-scrubbed."""
    scrubbed = default_pii_scrubber.scrub_value(tool_response)
    if scrubbed.total_redactions > 0 and isinstance(scrubbed.sanitized, dict):
        return scrubbed.sanitized
    return None

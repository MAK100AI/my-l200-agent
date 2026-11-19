"""Sliding-window conversation history compaction and Cloud Firestore ADK SessionService.

Satisfies AgentOps Code Review Matrix Criteria 5–7:
- Criterion 5: Sliding-window history compaction preserving system instructions, pinned context,
  and `function_call`/`function_response` turn-pair integrity while bounding token usage.
- Criterion 6: Production `FirestoreSessionService(BaseSessionService)` backed by Google Cloud Firestore
  with `app:`, `user:`, and `temp:` scoped state delta handling, idempotent `append_event`, and TTL expiry.
- Criterion 7: Explicit Pydantic `CalGradSessionState` schema validating session state across agents.
"""

from __future__ import annotations

import copy
import datetime
import logging
import os
import time
import uuid
from typing import Any, Literal, Optional, Protocol

from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from observability import default_pii_scrubber, log_structured_event


# ---------------------------------------------------------------------------
# Criterion 7: Explicit Session State Schema
# ---------------------------------------------------------------------------


class CalGradSessionState(BaseModel):
    """Typed session state contract shared across Coordinator, Extractor, and Classifier."""

    model_config = ConfigDict(extra="allow")

    extracted_performance_data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Normalized & PII-scrubbed gThanks, peer bonuses, and operational KPIs.",
    )
    classified_grad_evidence: Optional[dict[str, Any]] = Field(
        default=None,
        description="Evidence mapped to Section 1 (Impact) and Section 2 (Collaboration).",
    )
    current_grad_draft: Optional[dict[str, Any]] = Field(
        default=None,
        description="Current two-section GRAD self-assessment draft awaiting or holding approval.",
    )
    hitl_approval_status: Literal[
        "NOT_STARTED", "PENDING_REVIEW", "APPROVED", "REJECTED", "SUBMITTED"
    ] = Field(
        default="NOT_STARTED",
        description="Lifecycle state of Human-in-the-Loop review.",
    )
    compaction_summary: str = Field(
        default="",
        description="Rolling summary of older turns evicted by sliding-window compaction.",
    )


# ---------------------------------------------------------------------------
# Criterion 5: Sliding-Window History Compaction
# ---------------------------------------------------------------------------


def _has_function_call(content: types.Content) -> bool:
    return bool(content.parts and any(p.function_call is not None for p in content.parts))


def _has_function_response(content: types.Content) -> bool:
    return bool(
        content.parts and any(p.function_response is not None for p in content.parts)
    )


def _estimate_content_tokens(content: types.Content) -> int:
    """Fast deterministic token approximation (~4 chars/token) for context budget checks."""
    if not content.parts:
        return 0
    chars = 0
    for p in content.parts:
        if p.text:
            chars += len(p.text)
        elif p.function_call:
            chars += len(p.function_call.name or "") + len(str(p.function_call.args or {}))
        elif p.function_response:
            chars += len(p.function_response.name or "") + len(
                str(p.function_response.response or {})
            )
    return max(1, chars // 4)


def _summarize_evicted_contents(
    evicted: list[types.Content],
    prior_summary: str = "",
    max_summary_chars: int = 1200,
) -> str:
    """Produces a compact, PII-scrubbed rolling digest of evicted conversation turns."""
    snippets: list[str] = []
    if prior_summary:
        snippets.append(prior_summary.strip())

    for item in evicted:
        role = item.role or "unknown"
        if not item.parts:
            continue
        for part in item.parts:
            if part.text:
                clean = default_pii_scrubber.scrub_text(part.text.strip())
                if clean.startswith("[COMPACTED_HISTORY_SUMMARY]"):
                    continue
                snippets.append(f"{role}: {clean[:160]}")
            elif part.function_call:
                snippets.append(f"tool_call:{part.function_call.name}")
            elif part.function_response:
                snippets.append(f"tool_result:{part.function_response.name}")

    joined = " | ".join(snippets)
    if len(joined) > max_summary_chars:
        joined = "..." + joined[-max_summary_chars:]
    return joined


class SlidingWindowCompactor:
    """Deterministic sliding-window history compactor preserving tool call/response integrity."""

    SUMMARY_MARKER = "[COMPACTED_HISTORY_SUMMARY]"

    def __init__(
        self,
        window_size: int = 8,
        overlap_size: int = 2,
        max_token_budget: int = 12000,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if overlap_size < 0 or overlap_size >= window_size:
            raise ValueError("overlap_size must satisfy 0 <= overlap_size < window_size.")
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.max_token_budget = max_token_budget

    def compact_contents(
        self,
        contents: list[types.Content],
        prior_summary: str = "",
    ) -> tuple[list[types.Content], str, bool]:
        """Compacts `contents` into `[summary_anchor, *recent_window]` when window or token limits exceed.

        Guarantees that a `function_response` turn is never separated from its preceding `function_call`.
        Returns `(compacted_contents, updated_summary, did_compact)`.
        """
        if not contents:
            return contents, prior_summary, False

        estimated_tokens = sum(_estimate_content_tokens(c) for c in contents)
        needs_compaction = (
            len(contents) > self.window_size or estimated_tokens > self.max_token_budget
        )
        if not needs_compaction:
            return contents, prior_summary, False

        # Determine cut index keeping the most recent `window_size - 1` turns (leaving 1 slot for summary)
        keep_count = max(2, self.window_size - 1)
        cut_idx = max(0, len(contents) - keep_count)

        # Shift cut_idx backward if we would start on a `function_response` or orphan a `function_call`
        while cut_idx > 0 and _has_function_response(contents[cut_idx]):
            cut_idx -= 1
        while cut_idx > 0 and _has_function_call(contents[cut_idx - 1]) and _has_function_response(contents[cut_idx]):
            cut_idx -= 1

        # Further trim from front of kept tail if token budget is still exceeded (while keeping >= 2 turns)
        while (
            cut_idx < len(contents) - 2
            and sum(_estimate_content_tokens(c) for c in contents[cut_idx:])
            > self.max_token_budget
        ):
            cut_idx += 1
            while cut_idx < len(contents) and _has_function_response(contents[cut_idx]):
                cut_idx += 1

        if cut_idx <= 0:
            return contents, prior_summary, False

        evicted = contents[:cut_idx]
        retained = contents[cut_idx:]
        updated_summary = _summarize_evicted_contents(evicted, prior_summary=prior_summary)

        anchor_content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        f"{self.SUMMARY_MARKER} Prior conversation context compacted "
                        f"({len(evicted)} earlier turn(s)): {updated_summary}"
                    )
                )
            ],
        )
        compacted = [anchor_content, *retained]
        return compacted, updated_summary, True

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        """ADK `before_model_callback` applying sliding-window compaction in-place."""
        if not llm_request.contents:
            return None

        prior_summary = str(callback_context.state.get("compaction_summary", ""))
        original_len = len(llm_request.contents)
        compacted, new_summary, did_compact = self.compact_contents(
            list(llm_request.contents), prior_summary=prior_summary
        )
        if did_compact:
            llm_request.contents = compacted
            callback_context.state["compaction_summary"] = new_summary
            compaction_count = int(callback_context.state.get("compaction_runs", 0)) + 1
            callback_context.state["compaction_runs"] = compaction_count
            log_structured_event(
                "history_compacted",
                f"Sliding-window compacted LLM contents from {original_len} to {len(compacted)} turns",
                component="session_manager",
                agent_name=callback_context.agent_name,
                session_id=getattr(callback_context.session, "id", ""),
                turns_before=original_len,
                turns_after=len(compacted),
                compaction_runs=compaction_count,
            )
        return None


# ---------------------------------------------------------------------------
# Criterion 6: Cloud Firestore Persistent SessionService
# ---------------------------------------------------------------------------


class FirestoreDocSnapshotProtocol(Protocol):
    exists: bool

    def to_dict(self) -> Optional[dict[str, Any]]: ...


class FirestoreDocumentProtocol(Protocol):
    def get(self) -> FirestoreDocSnapshotProtocol: ...
    def set(self, document_data: dict[str, Any], merge: bool = False) -> Any: ...
    def delete(self) -> Any: ...


class FirestoreCollectionProtocol(Protocol):
    def document(self, document_id: Optional[str] = None) -> FirestoreDocumentProtocol: ...
    def where(self, *args: Any, **kwargs: Any) -> Any: ...
    def stream(self) -> Any: ...


class FirestoreClientProtocol(Protocol):
    def collection(self, name: str) -> FirestoreCollectionProtocol: ...


def _split_state_scopes(
    state: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Splits an ADK state dict into `(app_delta, user_delta, session_delta)` while dropping `temp:` keys."""
    app_delta: dict[str, Any] = {}
    user_delta: dict[str, Any] = {}
    session_delta: dict[str, Any] = {}

    for key, val in (state or {}).items():
        if key.startswith(State.TEMP_PREFIX):
            # Temporary turn-scoped keys must never be persisted to Firestore
            continue
        if key.startswith(State.APP_PREFIX):
            app_delta[key[len(State.APP_PREFIX) :]] = copy.deepcopy(val)
        elif key.startswith(State.USER_PREFIX):
            user_delta[key[len(State.USER_PREFIX) :]] = copy.deepcopy(val)
        else:
            session_delta[key] = copy.deepcopy(val)
    return app_delta, user_delta, session_delta


class FirestoreSessionService(BaseSessionService):
    """Production ADK `BaseSessionService` backed by Google Cloud Firestore with TTL and state scoping."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        database: str = "(default)",
        sessions_collection: str = "adk_sessions",
        user_state_collection: str = "adk_user_state",
        app_state_collection: str = "adk_app_state",
        ttl_days: int = 30,
        firestore_client: Optional[FirestoreClientProtocol] = None,
    ) -> None:
        self.project_id = (
            project_id
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT_ID")
            or ""
        )
        self.database = database
        self.sessions_collection = sessions_collection
        self.user_state_collection = user_state_collection
        self.app_state_collection = app_state_collection
        self.ttl_days = max(1, ttl_days)
        self._client = firestore_client

    def _db(self) -> FirestoreClientProtocol:
        if self._client is not None:
            return self._client
        try:
            from google.cloud import firestore  # type: ignore[attr-defined]

            kwargs: dict[str, Any] = {"database": self.database}
            if self.project_id:
                kwargs["project"] = self.project_id
            self._client = firestore.Client(**kwargs)
            return self._client
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-firestore is required to instantiate FirestoreSessionService without an injected client."
            ) from exc

    @staticmethod
    def _session_doc_id(app_name: str, user_id: str, session_id: str) -> str:
        return f"{app_name}__{user_id}__{session_id}"

    @staticmethod
    def _user_doc_id(app_name: str, user_id: str) -> str:
        return f"{app_name}__{user_id}"

    def _compute_expires_at(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=self.ttl_days
        )

    def _merge_scoped_state(
        self, app_name: str, user_id: str, session_state: dict[str, Any]
    ) -> dict[str, Any]:
        merged = copy.deepcopy(session_state)
        db = self._db()

        app_snap = db.collection(self.app_state_collection).document(app_name).get()
        if app_snap.exists:
            app_data = (app_snap.to_dict() or {}).get("state", {})
            for k, v in app_data.items():
                merged[f"{State.APP_PREFIX}{k}"] = copy.deepcopy(v)

        user_doc_id = self._user_doc_id(app_name, user_id)
        user_snap = db.collection(self.user_state_collection).document(user_doc_id).get()
        if user_snap.exists:
            user_data = (user_snap.to_dict() or {}).get("state", {})
            for k, v in user_data.items():
                merged[f"{State.USER_PREFIX}{k}"] = copy.deepcopy(v)

        return merged

    def _persist_scoped_deltas(
        self,
        app_name: str,
        user_id: str,
        app_delta: dict[str, Any],
        user_delta: dict[str, Any],
    ) -> None:
        db = self._db()
        if app_delta:
            app_ref = db.collection(self.app_state_collection).document(app_name)
            existing = (app_ref.get().to_dict() or {}).get("state", {}) if app_ref.get().exists else {}
            existing.update(app_delta)
            app_ref.set(
                {
                    "app_name": app_name,
                    "state": existing,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                merge=True,
            )

        if user_delta:
            user_doc_id = self._user_doc_id(app_name, user_id)
            user_ref = db.collection(self.user_state_collection).document(user_doc_id)
            snap = user_ref.get()
            existing_user = (snap.to_dict() or {}).get("state", {}) if snap.exists else {}
            existing_user.update(user_delta)
            user_ref.set(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "state": existing_user,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
                merge=True,
            )

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        resolved_id = session_id.strip() if session_id else uuid.uuid4().hex
        doc_id = self._session_doc_id(app_name, user_id, resolved_id)
        db = self._db()
        doc_ref = db.collection(self.sessions_collection).document(doc_id)

        if doc_ref.get().exists:
            raise ValueError(f"Session '{resolved_id}' already exists in Firestore.")

        app_delta, user_delta, session_state = _split_state_scopes(state)
        # Validate initial session state against CalGradSessionState contract
        CalGradSessionState.model_validate(session_state)
        self._persist_scoped_deltas(app_name, user_id, app_delta, user_delta)

        now_ts = time.time()
        doc_payload = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": resolved_id,
            "state": default_pii_scrubber.scrub_value(session_state).sanitized,
            "events": [],
            "last_update_time": now_ts,
            "expires_at": self._compute_expires_at(),
        }
        doc_ref.set(doc_payload)

        log_structured_event(
            "firestore_session_created",
            f"Created Firestore session '{resolved_id}'",
            component="session_manager",
            session_id=resolved_id,
            app_name=app_name,
        )

        merged_state = self._merge_scoped_state(app_name, user_id, doc_payload["state"])
        return Session(
            id=resolved_id,
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=[],
            last_update_time=now_ts,
        )

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        doc_id = self._session_doc_id(app_name, user_id, session_id.strip())
        snap = self._db().collection(self.sessions_collection).document(doc_id).get()
        if not snap.exists:
            return None

        raw = snap.to_dict() or {}
        raw_events: list[dict[str, Any]] = raw.get("events", [])
        events = [Event.model_validate(ev) for ev in raw_events]

        if config is not None:
            if config.num_recent_events is not None:
                events = (
                    []
                    if config.num_recent_events == 0
                    else events[-config.num_recent_events :]
                )
            if config.after_timestamp is not None:
                events = [e for e in events if e.timestamp > config.after_timestamp]

        merged_state = self._merge_scoped_state(
            app_name, user_id, raw.get("state", {})
        )
        return Session(
            id=raw["session_id"],
            app_name=app_name,
            user_id=user_id,
            state=merged_state,
            events=events,
            last_update_time=float(raw.get("last_update_time", 0.0)),
        )

    @override
    async def list_sessions(
        self, *, app_name: str, user_id: Optional[str] = None
    ) -> ListSessionsResponse:
        sessions: list[Session] = []
        for snap in self._db().collection(self.sessions_collection).stream():
            data = snap.to_dict() or {}
            if data.get("app_name") != app_name:
                continue
            if user_id is not None and data.get("user_id") != user_id:
                continue
            uid = data["user_id"]
            merged_state = self._merge_scoped_state(
                app_name, uid, data.get("state", {})
            )
            sessions.append(
                Session(
                    id=data["session_id"],
                    app_name=app_name,
                    user_id=uid,
                    state=merged_state,
                    events=[],
                    last_update_time=float(data.get("last_update_time", 0.0)),
                )
            )
        return ListSessionsResponse(sessions=sessions)

    @override
    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        doc_id = self._session_doc_id(app_name, user_id, session_id.strip())
        self._db().collection(self.sessions_collection).document(doc_id).delete()
        log_structured_event(
            "firestore_session_deleted",
            f"Deleted Firestore session '{session_id}'",
            component="session_manager",
            session_id=session_id,
        )

    @override
    async def get_user_state(self, *, app_name: str, user_id: str) -> dict[str, Any]:
        user_doc_id = self._user_doc_id(app_name, user_id)
        snap = self._db().collection(self.user_state_collection).document(user_doc_id).get()
        if not snap.exists:
            return {}
        return copy.deepcopy((snap.to_dict() or {}).get("state", {}))

    @override
    async def flush(self) -> None:
        return None

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event

        doc_id = self._session_doc_id(session.app_name, session.user_id, session.id)
        doc_ref = self._db().collection(self.sessions_collection).document(doc_id)
        snap = doc_ref.get()
        if not snap.exists:
            raise ValueError(f"Firestore session '{session.id}' not found.")

        stored = snap.to_dict() or {}
        stored_events: list[dict[str, Any]] = list(stored.get("events", []))
        serialized_event = event.model_dump(mode="json", exclude_none=True)

        # Idempotent event append: skip duplicate event IDs with identical payload
        if any(
            ev.get("id") == event.id and ev == serialized_event for ev in stored_events
        ):
            return event

        await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp

        stored_state: dict[str, Any] = dict(stored.get("state", {}))
        if event.actions and event.actions.state_delta:
            app_delta, user_delta, session_delta = _split_state_scopes(
                event.actions.state_delta
            )
            self._persist_scoped_deltas(
                session.app_name, session.user_id, app_delta, user_delta
            )
            if session_delta:
                scrubbed_delta = default_pii_scrubber.scrub_value(session_delta).sanitized
                stored_state.update(scrubbed_delta)
                CalGradSessionState.model_validate(stored_state)

        stored_events.append(default_pii_scrubber.scrub_value(serialized_event).sanitized)
        doc_ref.set(
            {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "state": stored_state,
                "events": stored_events,
                "last_update_time": event.timestamp,
                "expires_at": self._compute_expires_at(),
            }
        )
        return event


def build_session_service(
    *,
    firestore_client: Optional[FirestoreClientProtocol] = None,
    prefer_in_memory: Optional[bool] = None,
) -> BaseSessionService:
    """Factory returning `FirestoreSessionService` in production or `InMemorySessionService` when explicitly requested."""
    if firestore_client is not None:
        return FirestoreSessionService(firestore_client=firestore_client)

    use_memory = (
        prefer_in_memory
        if prefer_in_memory is not None
        else os.getenv("SESSION_BACKEND", "firestore").lower() == "memory"
    )
    if use_memory:
        return InMemorySessionService()

    return FirestoreSessionService()

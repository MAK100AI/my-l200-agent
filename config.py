"""Application configuration and environment settings for the CAL GRAD ADK system."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

COORDINATOR_MODEL = "gemini-2.5-pro"
WORKER_MODEL = "gemini-2.5-flash"
APP_NAME = "cal_grad_performance_app"


class AgentOpsSettings(BaseSettings):
    """Validated runtime configuration for the 3-agent CAL GRAD system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "test", "staging", "production"] = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Deployment environment controlling strictness of secrets and storage.",
    )
    project_id: str = Field(
        default="cal-grad-agentops-dev",
        alias="GOOGLE_CLOUD_PROJECT",
        min_length=4,
        description="Google Cloud Project ID.",
    )
    location: str = Field(
        default="us-central1",
        alias="GOOGLE_CLOUD_LOCATION",
        description="Google Cloud region for Vertex AI and Model Armor.",
    )
    use_vertexai: bool = Field(
        default=True,
        alias="GOOGLE_GENAI_USE_VERTEXAI",
        description="Route google-genai calls through Vertex AI.",
    )

    # Model tiering (AgentOps Criterion: Pro for Coordinator/Formatting, Flash for Extract/Classify)
    coordinator_model: str = Field(
        default=COORDINATOR_MODEL,
        alias="CAL_COORDINATOR_MODEL",
        description="Model for complex orchestration and GRAD narrative formatting.",
    )
    worker_model: str = Field(
        default=WORKER_MODEL,
        alias="CAL_WORKER_MODEL",
        description="Low-latency model for structured extraction and rubric classification.",
    )
    max_llm_calls_per_run: int = Field(
        default=20,
        ge=3,
        le=100,
        alias="MAX_LLM_CALLS_PER_RUN",
        description="Circuit-breaker cap on LLM calls per invocation.",
    )

    # Firestore & sliding-window compaction settings
    use_firestore_sessions: bool = Field(
        default=False,
        alias="USE_FIRESTORE_SESSIONS",
        description="Enable FirestoreSessionService; defaults to True automatically in staging/production.",
    )
    firestore_database: str = Field(
        default="(default)",
        alias="FIRESTORE_DATABASE",
        description="Firestore Native database identifier.",
    )
    firestore_root_collection: str = Field(
        default="cal-grad-adk-sessions",
        alias="ADK_FIRESTORE_ROOT_COLLECTION",
        description="Root Firestore collection for ADK session state and events.",
    )
    session_ttl_seconds: int = Field(
        default=60 * 60 * 24 * 30,
        ge=300,
        alias="SESSION_TTL_SECONDS",
        description="TTL retention window (seconds) for persisted session documents.",
    )
    compaction_interval: int = Field(
        default=4,
        ge=1,
        le=50,
        alias="COMPACTION_INTERVAL",
        description="Number of new invocations triggering sliding-window compaction.",
    )
    compaction_overlap_size: int = Field(
        default=1,
        ge=0,
        le=10,
        alias="COMPACTION_OVERLAP_SIZE",
        description="Overlap invocations retained across sliding-window summaries.",
    )
    compaction_token_threshold: int = Field(
        default=16_000,
        ge=1_000,
        alias="COMPACTION_TOKEN_THRESHOLD",
        description="Prompt token threshold triggering post-invocation event compaction.",
    )
    compaction_event_retention_size: int = Field(
        default=4,
        ge=1,
        alias="COMPACTION_EVENT_RETENTION_SIZE",
        description="Number of raw recent events retained after compaction.",
    )
    sliding_window_max_contents: int = Field(
        default=12,
        ge=4,
        le=100,
        alias="SLIDING_WINDOW_MAX_CONTENTS",
        description="Hard sliding-window cap on LlmRequest contents turns.",
    )

    # Model Armor guardrails & Human-in-the-Loop settings
    enable_model_armor: bool = Field(
        default=True,
        alias="ENABLE_MODEL_ARMOR",
        description="Enable Model Armor prompt/response/tool guardrails.",
    )
    model_armor_template_id: str = Field(
        default="cal-grad-guardrail-template",
        alias="MODEL_ARMOR_TEMPLATE_ID",
        description="Model Armor template ID or full resource path.",
    )
    model_armor_fail_closed: bool = Field(
        default=True,
        alias="MODEL_ARMOR_FAIL_CLOSED",
        description="Block execution when Model Armor screening fails or detects a threat.",
    )
    require_hitl_approval: bool = Field(
        default=True,
        alias="REQUIRE_HITL_APPROVAL",
        description="Require explicit CAL human confirmation before GRAD submission.",
    )

    # Secret Manager & logging settings
    allow_env_secret_fallback: bool = Field(
        default=True,
        alias="ALLOW_ENV_SECRET_FALLBACK",
        description="Allow environment variable fallback in non-production environments.",
    )
    secret_cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        alias="SECRET_CACHE_TTL_SECONDS",
        description="In-memory TTL cache duration for fetched secrets.",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Minimum structured JSON logging level.",
    )

    @model_validator(mode="after")
    def _enforce_production_defaults(self) -> AgentOpsSettings:
        if self.environment in ("staging", "production"):
            self.use_firestore_sessions = True
        if self.environment == "production":
            self.allow_env_secret_fallback = False
            self.model_armor_fail_closed = True
            self.require_hitl_approval = True
        if self.compaction_overlap_size >= self.compaction_interval:
            raise ValueError(
                "compaction_overlap_size must be strictly less than compaction_interval."
            )
        return self

    @property
    def model_armor_template_resource(self) -> str:
        """Returns the fully qualified Model Armor template resource path."""
        if self.model_armor_template_id.startswith("projects/"):
            return self.model_armor_template_id
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/templates/{self.model_armor_template_id}"
        )


@lru_cache(maxsize=1)
def get_settings() -> AgentOpsSettings:
    """Returns cached singleton application settings."""
    return AgentOpsSettings()

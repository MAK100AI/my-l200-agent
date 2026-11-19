variable "project_id" {
  description = "Google Cloud Project ID hosting the CAL GRAD ADK multi-agent workload."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) >= 6
    error_message = "project_id must be a valid non-empty GCP project identifier."
  }
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Firestore, and Model Armor."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment tier (dev, staging, production)."
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be one of: dev, staging, production."
  }
}

variable "firestore_database_id" {
  description = "Firestore Native database identifier storing ADK sessions and scoped state."
  type        = string
  default     = "(default)"
}

variable "model_armor_template_id" {
  description = "Identifier for the Google Cloud Model Armor template enforcing prompt/output guardrails."
  type        = string
  default     = "cal-grad-armor-template"
}

variable "container_image" {
  description = "Artifact Registry container image digest/tag for the CAL GRAD ADK service."
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/agentops/cal-grad-agent:latest"
}

variable "log_retention_days" {
  description = "Retention period in days for structured JSON audit and telemetry logs."
  type        = number
  default     = 90
}

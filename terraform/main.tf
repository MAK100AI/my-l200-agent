locals {
  required_services = toset([
    "aiplatform.googleapis.com",
    "firestore.googleapis.com",
    "secretmanager.googleapis.com",
    "modelarmor.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "run.googleapis.com",
  ])

  common_labels = {
    application = "cal-grad-performance-agent"
    environment = var.environment
    managed_by  = "terraform"
    framework   = "google-adk"
  }
}

resource "google_project_service" "required_apis" {
  for_each = local.required_services

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# 1. Cloud Firestore Native Database & ADK Session TTL Policy
# ---------------------------------------------------------------------------

resource "google_firestore_database" "adk_sessions_db" {
  project                     = var.project_id
  name                        = var.firestore_database_id
  location_id                 = var.region
  type                        = "FIRESTORE_NATIVE"
  concurrency_mode            = "OPTIMISTIC"
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"
  delete_protection_state     = "DELETE_PROTECTION_ENABLED"

  depends_on = [google_project_service.required_apis]
}

resource "google_firestore_field" "adk_sessions_ttl" {
  project    = var.project_id
  database   = google_firestore_database.adk_sessions_db.name
  collection = "adk_sessions"
  field      = "expires_at"

  ttl_config {}
}

# ---------------------------------------------------------------------------
# 2. Google Cloud Secret Manager Secrets
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret" "grad_hmac_signing_key" {
  project   = var.project_id
  secret_id = "cal-grad-hmac-signing-key"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "model_armor_template_id" {
  project   = var.project_id
  secret_id = "model-armor-template-id"
  labels    = local.common_labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_version" "model_armor_template_id_v1" {
  secret      = google_secret_manager_secret.model_armor_template_id.id
  secret_data = google_model_armor_template.cal_grad_guardrail.template_id
}

# ---------------------------------------------------------------------------
# 3. Google Cloud Model Armor Guardrail Template
# ---------------------------------------------------------------------------

resource "google_model_armor_template" "cal_grad_guardrail" {
  provider    = google-beta
  project     = var.project_id
  location    = var.region
  template_id = var.model_armor_template_id
  labels      = local.common_labels

  filter_config {
    rai_settings {
      rai_filters {
        filter_type      = "HATE_SPEECH"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "HARASSMENT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "DANGEROUS"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
      rai_filters {
        filter_type      = "SEXUALLY_EXPLICIT"
        confidence_level = "MEDIUM_AND_ABOVE"
      }
    }

    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "HIGH"
    }

    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }

    sdp_settings {
      basic_config {
        filter_enforcement = "ENABLED"
      }
    }
  }

  template_metadata {
    log_template_operations = true
    log_sanitize_operations = true
  }

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# 4. Observability: Log-Based Guardrail & PII Telemetry Metrics
# ---------------------------------------------------------------------------

resource "google_logging_metric" "guardrail_blocks_metric" {
  project     = var.project_id
  name        = "cal_grad_agent/guardrail_blocks_count"
  description = "Counts Model Armor prompt-injection and unsafe output blocks across CAL GRAD agents."
  filter      = "jsonPayload.component=\"guardrails\" AND (jsonPayload.event_type=\"guardrail_blocked_input\" OR jsonPayload.event_type=\"guardrail_blocked_output\")"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }

  depends_on = [google_project_service.required_apis]
}

# ---------------------------------------------------------------------------
# 5. Cloud Run v2 Agent Runtime Service
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "cal_grad_agent_service" {
  project             = var.project_id
  name                = "cal-grad-agent-${var.environment}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  deletion_protection = true
  labels              = local.common_labels

  template {
    service_account                  = google_service_account.cal_grad_agent.email
    max_instance_request_concurrency = 20
    timeout                          = "120s"

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "APP_ENV"
        value = var.environment
      }
      env {
        name  = "SESSION_BACKEND"
        value = "firestore"
      }
      env {
        name  = "MODEL_ARMOR_REMOTE_ENABLED"
        value = "true"
      }
      env {
        name  = "MODEL_ARMOR_FAIL_CLOSED"
        value = "true"
      }
      env {
        name  = "MODEL_ARMOR_LOCATION"
        value = var.region
      }
      env {
        name = "MODEL_ARMOR_TEMPLATE_ID"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.model_armor_template_id.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_iam_member.agent_least_privilege_roles,
    google_secret_manager_secret_version.model_armor_template_id_v1,
  ]
}

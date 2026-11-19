# Least-privilege Service Account and resource-scoped IAM bindings (Criterion 18)

resource "google_service_account" "cal_grad_agent" {
  account_id   = "cal-grad-agent-${var.environment}"
  display_name = "CAL GRAD Multi-Agent Runtime Service Account (${var.environment})"
  description  = "Least-privilege identity for the 3-agent ADK CAL GRAD performance system."
}

locals {
  agent_project_roles = toset([
    "roles/aiplatform.user",
    "roles/datastore.user",
    "roles/modelarmor.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ])
}

resource "google_project_iam_member" "agent_least_privilege_roles" {
  for_each = local.agent_project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cal_grad_agent.email}"

  depends_on = [google_project_service.required_apis]
}

# Resource-level Secret Manager IAM (never grant project-wide secretAccessor)
resource "google_secret_manager_secret_iam_member" "hmac_signing_key_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.grad_hmac_signing_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cal_grad_agent.email}"
}

resource "google_secret_manager_secret_iam_member" "model_armor_template_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.model_armor_template_id.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cal_grad_agent.email}"
}

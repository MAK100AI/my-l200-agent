output "service_account_email" {
  description = "Least-privilege service account email bound to the CAL GRAD ADK runtime."
  value       = google_service_account.cal_grad_agent.email
}

output "firestore_database_name" {
  description = "Firestore Native database storing persistent ADK sessions."
  value       = google_firestore_database.adk_sessions_db.name
}

output "model_armor_template_name" {
  description = "Full resource name of the provisioned Google Cloud Model Armor guardrail template."
  value       = google_model_armor_template.cal_grad_guardrail.name
}

output "cloud_run_service_uri" {
  description = "Internal HTTPS URI of the deployed CAL GRAD ADK Cloud Run v2 service."
  value       = google_cloud_run_v2_service.cal_grad_agent_service.uri
}

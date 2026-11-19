provider "google" {
  project = "adeelkl200"
  region  = "us-central1"
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "GEMINI_API_KEY"
  replication {
    automatic = true
  }
}

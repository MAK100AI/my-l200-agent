import os
from google.cloud import secretmanager

def get_gemini_api_key() -> str:
    """Retrieves the GEMINI_API_KEY from Secret Manager safely."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/adeelkl200/secrets/GEMINI_API_KEY/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "mock_key_fallback")

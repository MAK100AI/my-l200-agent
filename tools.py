from pydantic import BaseModel, Field, Extra
import structlog

logger = structlog.get_logger()

class ValidateInputs(BaseModel):
    name: str = Field(..., description="Name of the direct report")
    class Config:
        extra = Extra.forbid

def validate_domain_inputs(name: str) -> str:
    """Validates inputs before processing GRAD structures."""
    try:
        if not name:
            raise ValueError("Name cannot be empty")
        return f"Validated: {name}"
    except Exception as e:
        logger.error("tool_error", error=str(e))
        return f"Recovery Hint: Please provide a valid, non-empty string name."

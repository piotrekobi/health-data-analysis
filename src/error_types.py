from enum import Enum
from typing import Optional, Dict

class HealthQueryErrorType(Enum):
    QUERY_GENERATION = "query_generation"
    DATA_VALIDATION = "data_validation"
    DATABASE_ERROR = "database_error"
    MODEL_ERROR = "model_error"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"

class HealthQueryError:
    ERROR_MESSAGES = {
        HealthQueryErrorType.QUERY_GENERATION: (
            "I couldn't generate a valid SQL query for your question. "
            "Please try rephrasing your question or provide more specific details."
        ),
        HealthQueryErrorType.DATA_VALIDATION: (
            "The data format or size is invalid. "
            "Please try a different question or add filters to reduce the data size."
        ),
        HealthQueryErrorType.DATABASE_ERROR: (
            "There was an error accessing or processing the data. "
            "Please try again or use different parameters."
        ),
        HealthQueryErrorType.MODEL_ERROR: (
            "There was an error processing your request with the AI model. "
            "Please try again or simplify your question."
        ),
        HealthQueryErrorType.SERVER_ERROR: (
            "The server encountered an unexpected error. "
            "Please try again later or contact support."
        ),
        HealthQueryErrorType.CONNECTION_ERROR: (
            "Could not connect to the server. "
            "Please check your internet connection and try again."
        ),
        HealthQueryErrorType.TIMEOUT_ERROR: (
            "The request took too long to complete. "
            "Please try a simpler question or try again later."
        )
    }

    @classmethod
    def format_error(cls, error_type: HealthQueryErrorType, detail: Optional[str] = None) -> Dict:
        error_msg = cls.ERROR_MESSAGES[error_type]
        return {
            "type": "error",
            "errorType": error_type.value,
            "error": error_msg,
            "detail": detail
        }
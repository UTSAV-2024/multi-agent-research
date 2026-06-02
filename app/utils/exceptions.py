class AppException(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "APP_ERROR"
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ResearchException(AppException):
    def __init__(
        self,
        message: str = "Research pipeline failed"
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RESEARCH_ERROR"
        )


class LLMException(AppException):
    def __init__(
        self,
        message: str = "LLM execution failed"
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="LLM_ERROR"
        )


class RetrievalException(AppException):
    def __init__(
        self,
        message: str = "Retrieval failed"
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RETRIEVAL_ERROR"
        )
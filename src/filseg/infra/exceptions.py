"""Project-level exceptions."""


class FilsegError(Exception):
    """Base class for all filseg errors."""


class SubmissionValidationError(FilsegError):
    """Raised when a submission file violates the competition format."""

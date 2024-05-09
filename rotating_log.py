import logging
import os
from logging.handlers import RotatingFileHandler


class RotatingLogFile:
    """
    A logging utility class that supports log rotation and conditional logging to both
    file and standard output based on the log level. Logs with level DEBUG are written
    to the file only, while logs with level INFO or higher are also echoed to the console.
    """

    def __init__(
        self,
        logger_name,
        log_filename,
        max_bytes,
        backup_count,
        log_level="INFO",
    ):
        """
        Initializes the class with specified parameters and configures logging to the
        console and file with rotation based on log level.

        Args:
        - logger_name (str): The name for the logger.
        - log_filename (str): Path to the log file.
        - max_bytes (int): Maximum size of the log file in bytes before it's rotated.
        - backup_count (int): The number of backup files to retain.
        - log_level (str): The log level to capture, specified as a string (e.g., "DEBUG", "INFO").
        """
        self.log_filename = log_filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Convert log level from string to logging level
        str_to_log_level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        numeric_level = str_to_log_level.get(log_level.upper(), logging.INFO)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(
            logging.DEBUG
        )  # Capture all logs at the logger level

        # Create the file if it doesn't exist
        if not os.path.exists(log_filename):
            open(log_filename, "a").close()

        # Setup log file format
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        # Setup the RotatingFileHandler for file logging with DEBUG level
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # This handler captures all logs
        self.logger.addHandler(file_handler)

        # Setup the StreamHandler for console logging with INFO level
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(
            logging.INFO
        )  # This handler captures only INFO or higher level logs
        self.logger.addHandler(console_handler)

    def debug(self, message):
        """Logs a debug message."""
        self.logger.debug(message)

    def info(self, message):
        """Logs an info message."""
        self.logger.info(message)

    def warning(self, message):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Logs an error message."""
        self.logger.error(message)

    def critical(self, message):
        """Logs a critical message."""
        self.logger.critical(message)


# Example usage
if __name__ == "__main__":
    log_file = "sample.log"
    logger_name = "exampleLogger"
    max_bytes = 1024 * 1024 * 5  # 5 MB
    backup_count = 3  # Keep 3 backup files

    logger = RotatingLogFile(
        logger_name, log_file, max_bytes, backup_count, log_level="DEBUG"
    )
    logger.debug(
        "This is a debug message."
    )  # This will only appear in the log file
    logger.info("This is an info message.")  # This will appear both in the

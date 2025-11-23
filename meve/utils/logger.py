"""
MeVe Framework - Production-Level Logging Utility

Features:
- Multiple log levels (debug, info, warn, error, critical)
- Color-coded console output for better DX
- Structured logging with context/metadata
- Performance timing utilities
- Error tracking with stack traces
- Production-safe (respects environment)
- Easy integration with external logging services (Sentry, etc.)
- Scoped loggers with prefixes
- Log buffering for error reporting
- Network/lifecycle event logging
"""

import logging
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from enum import IntEnum
from datetime import datetime
import yaml


class LogLevel(IntEnum):
    """Log level enumeration."""

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50
    NONE = 100


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for beautiful console output."""

    RESET = "\033[0m"
    BRIGHT = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Background colors
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"


# Emoji indicators for better visual feedback
class Indicators:
    """Unicode indicators for different log types."""

    DEBUG = "🔍"
    INFO = "ℹ️ "
    WARNING = "⚠️ "
    ERROR = "❌"
    CRITICAL = "🔥"
    SUCCESS = "✅"
    TIME = "⏱️ "
    ROCKET = "🚀"
    NETWORK = "🌐"
    NAVIGATION = "🧭"
    LIFECYCLE = "♻️ "
    DIVIDER = "─"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support."""

    def __init__(self, use_colors: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors
        # Store the original format to avoid accumulation
        self._original_fmt = self._style._fmt

        self.level_colors = {
            logging.DEBUG: Colors.GRAY,
            logging.INFO: Colors.BLUE,
            logging.WARNING: Colors.YELLOW,
            logging.ERROR: Colors.RED,
            logging.CRITICAL: f"{Colors.BG_RED}{Colors.WHITE}{Colors.BRIGHT}",
        }

        self.level_indicators = {
            logging.DEBUG: Indicators.DEBUG,
            logging.INFO: Indicators.INFO,
            logging.WARNING: Indicators.WARNING,
            logging.ERROR: Indicators.ERROR,
            logging.CRITICAL: Indicators.CRITICAL,
        }

    def format(self, record):
        if self.use_colors:
            # Add color to level name
            levelname = record.levelname
            color = self.level_colors.get(record.levelno, "")
            indicator = self.level_indicators.get(record.levelno, "")

            record.levelname = f"{indicator} {color}{Colors.BRIGHT}[{levelname}]{Colors.RESET}"
            record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"

            # Color the timestamp - use original format to avoid accumulation
            self._style._fmt = f"{Colors.DIM}%(asctime)s{Colors.RESET} {self._original_fmt.replace('%(asctime)s - ', '')}"

        return super().format(record)


class Logger:
    """
    Enhanced logger instance with scoping, timing, and metadata support.

    This is the instance class returned by get_logger(). It wraps
    Python's logging.Logger with additional features.
    """

    def __init__(
        self, logger: logging.Logger, prefix: str = "", context: Optional[Dict[str, Any]] = None
    ):
        self._logger = logger
        self._prefix = prefix
        self._context = context or {}
        self._timers: Dict[str, float] = {}

    def _format_metadata(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format metadata for logging."""
        combined = {**self._context, **(metadata or {})}
        if not combined:
            return ""
        try:
            return f"\n{json.dumps(combined, indent=2, default=str)}"
        except Exception:
            return f"\n{str(combined)}"

    def _log(
        self,
        level: int,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None,
    ):
        """Internal logging method."""
        full_message = f"{self._prefix}{message}" if self._prefix else message
        meta_str = self._format_metadata(metadata)

        # Add to buffer
        MeVeLogger._add_to_buffer(level, full_message, {**self._context, **(metadata or {})})

        # Log the message
        self._logger.log(level, full_message + meta_str, exc_info=exc_info)

    def debug(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self._log(logging.DEBUG, message, metadata)

    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self._log(logging.INFO, message, metadata)

    def warn(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self._log(logging.WARNING, message, metadata)

    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Alias for warn()."""
        self.warn(message, metadata)

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log error message with optional exception."""
        enriched_metadata = metadata or {}
        if error:
            enriched_metadata["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc() if MeVeLogger._enable_stack_trace else None,
            }
        self._log(
            logging.ERROR,
            message,
            enriched_metadata,
            exc_info=error if MeVeLogger._enable_stack_trace else None,
        )

    def critical(
        self,
        message: str,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log critical message."""
        enriched_metadata = metadata or {}
        if error:
            enriched_metadata["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc() if MeVeLogger._enable_stack_trace else None,
            }
        self._log(
            logging.CRITICAL,
            message,
            enriched_metadata,
            exc_info=error if MeVeLogger._enable_stack_trace else None,
        )

    def success(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log success message (info level with success indicator)."""
        self.info(f"{Indicators.SUCCESS} {message}", metadata)

    def divider(self, label: Optional[str] = None):
        """Print a visual divider."""
        line = Indicators.DIVIDER * 50
        if label:
            self.info(f"{Colors.CYAN}{line} {label} {line}{Colors.RESET}")
        else:
            self.info(f"{Colors.GRAY}{line}{Colors.RESET}")

    def time(self, label: str):
        """Start a performance timer."""
        self._timers[label] = time.time()
        self.debug(f"{Indicators.TIME}Timer started: {label}")

    def timeEnd(self, label: str) -> Optional[float]:
        """End a performance timer and log duration."""
        if label not in self._timers:
            self.warn(f"Timer '{label}' does not exist")
            return None

        duration = time.time() - self._timers[label]
        del self._timers[label]

        # Color code based on duration
        if duration < 0.1:
            color = Colors.GREEN
        elif duration < 1.0:
            color = Colors.YELLOW
        else:
            color = Colors.RED

        self.info(
            f"{Indicators.TIME}{color}Timer '{label}': {duration * 1000:.2f}ms{Colors.RESET}",
            {"duration_ms": duration * 1000, "label": label},
        )
        return duration

    def lifecycle(self, event: str, metadata: Optional[Dict[str, Any]] = None):
        """Log lifecycle event."""
        self.info(f"{Indicators.LIFECYCLE}Lifecycle: {event}", metadata)

    def network(self, method: str, url: str, metadata: Optional[Dict[str, Any]] = None):
        """Log network request."""
        method_colors = {
            "GET": Colors.GREEN,
            "POST": Colors.BLUE,
            "PUT": Colors.YELLOW,
            "DELETE": Colors.RED,
            "PATCH": Colors.MAGENTA,
        }
        color = method_colors.get(method.upper(), Colors.WHITE)
        self.debug(f"{Indicators.NETWORK} {color}{method}{Colors.RESET} {url}", metadata)

    def navigation(self, location: str, params: Optional[Dict[str, Any]] = None):
        """Log navigation event."""
        self.debug(f"{Indicators.NAVIGATION}Navigation: {location}", {"params": params})

    def table(self, data: Any):
        """Log data as a table (development only)."""
        if MeVeLogger._log_level <= logging.DEBUG:
            print("\n" + "=" * 60)
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"{key:30} | {value}")
            elif isinstance(data, (list, tuple)):
                for item in data:
                    print(f"  • {item}")
            print("=" * 60 + "\n")

    def group(self, label: str):
        """Start a log group."""
        self.info(f"{Colors.CYAN}{Colors.BRIGHT}┌─ {label} ─┐{Colors.RESET}")

    def groupEnd(self, label: str = ""):
        """End a log group."""
        end_label = f" {label} " if label else ""
        self.info(f"{Colors.CYAN}{Colors.BRIGHT}└─{end_label}─┘{Colors.RESET}")

    def assert_true(self, condition: bool, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Assert a condition and log error if false."""
        if not condition:
            self.error(f"Assertion failed: {message}", Exception(message), metadata)

    def scope(self, prefix: str) -> "Logger":
        """Create a scoped logger with a prefix."""
        new_prefix = f"{self._prefix}[{prefix}] " if self._prefix else f"[{prefix}] "
        return Logger(self._logger, prefix=new_prefix, context=self._context.copy())

    def child(self, context: Dict[str, Any]) -> "Logger":
        """Create a child logger with persistent context."""
        merged_context = {**self._context, **context}
        return Logger(self._logger, prefix=self._prefix, context=merged_context)


class MeVeLogger:
    """
    Production-level logger manager for MeVe framework.

    This is a singleton class that manages logger configuration and instances.
    """

    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    _log_level = LogLevel.INFO
    _log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _log_file: Optional[Path] = None
    _console_handler: Optional[logging.Handler] = None
    _file_handler: Optional[logging.Handler] = None
    _use_colors = True
    _enable_timestamps = True
    _enable_stack_trace = True
    _log_buffer: List[Dict[str, Any]] = []
    _buffer_size = 100
    _external_handler: Optional[Callable] = None

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_format: Optional[str] = None,
        log_file: Optional[str] = None,
        console_output: bool = True,
        use_colors: bool = True,
        enable_timestamps: bool = True,
        enable_stack_trace: bool = True,
        buffer_size: int = 100,
    ):
        """
        Configure the global logging settings for MeVe.

        Parameters
        ----------
        level : str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format : str, optional
            Custom format string for log messages
        log_file : str, optional
            Path to log file. If provided, logs will be written to file.
        console_output : bool
            Whether to output logs to console (default: True)
        use_colors : bool
            Whether to use colored output (default: True)
        enable_timestamps : bool
            Whether to include timestamps (default: True)
        enable_stack_trace : bool
            Whether to include stack traces for errors (default: True)
        buffer_size : int
            Maximum number of logs to keep in buffer (default: 100)
        """
        cls._log_level = getattr(logging, level.upper(), logging.INFO)
        cls._use_colors = use_colors
        cls._enable_timestamps = enable_timestamps
        cls._enable_stack_trace = enable_stack_trace
        cls._buffer_size = buffer_size

        if log_format:
            cls._log_format = log_format

        # Create formatters
        if use_colors and console_output:
            console_formatter = ColoredFormatter(
                use_colors=True, fmt=cls._log_format, datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            console_formatter = logging.Formatter(fmt=cls._log_format, datefmt="%Y-%m-%d %H:%M:%S")

        file_formatter = logging.Formatter(fmt=cls._log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler
        if console_output:
            cls._console_handler = logging.StreamHandler(sys.stdout)
            cls._console_handler.setLevel(cls._log_level)
            cls._console_handler.setFormatter(console_formatter)

        # File handler
        if log_file:
            cls._log_file = Path(log_file)
            cls._log_file.parent.mkdir(parents=True, exist_ok=True)
            cls._file_handler = logging.FileHandler(cls._log_file)
            cls._file_handler.setLevel(cls._log_level)
            cls._file_handler.setFormatter(file_formatter)

        cls._configured = True

        # Update existing loggers
        for logger_name in cls._loggers:
            cls._apply_handlers(cls._loggers[logger_name])

    @classmethod
    def configure_from_yaml(cls, config_path: str, env: str = "development"):
        """
        Load logging configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        env : str
            Environment name (development/production)
        """
        config_file = Path(config_path)

        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            cls.configure()
            return

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logging_config = config.get("logging", {})

        cls.configure(
            level=logging_config.get("level", "INFO"),
            log_format=logging_config.get("format"),
            log_file=logging_config.get("file"),
            console_output=True,
            use_colors=logging_config.get("use_colors", True),
            enable_timestamps=logging_config.get("enable_timestamps", True),
            enable_stack_trace=logging_config.get("enable_stack_trace", True),
        )

    @classmethod
    def _apply_handlers(cls, logger: logging.Logger):
        """Apply configured handlers to a logger."""
        # Remove existing handlers
        logger.handlers.clear()

        # Add console handler
        if cls._console_handler:
            logger.addHandler(cls._console_handler)

        # Add file handler
        if cls._file_handler:
            logger.addHandler(cls._file_handler)

        logger.setLevel(cls._log_level)
        logger.propagate = False

    @classmethod
    def get_logger(cls, name: str) -> Logger:
        """
        Get or create a logger for a specific module.

        Parameters
        ----------
        name : str
            Logger name (typically __name__ of the module)

        Returns
        -------
        Logger
            Enhanced logger instance with scoping and timing support
        """
        if name in cls._loggers:
            return Logger(cls._loggers[name])

        # Auto-configure if not done yet
        if not cls._configured:
            cls.configure()

        logger = logging.getLogger(name)
        cls._apply_handlers(logger)

        cls._loggers[name] = logger
        return Logger(logger)

    @classmethod
    def set_level(cls, level: str):
        """
        Dynamically change the logging level for all loggers.

        Parameters
        ----------
        level : str
            New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        cls._log_level = getattr(logging, level.upper(), logging.INFO)

        if cls._console_handler:
            cls._console_handler.setLevel(cls._log_level)

        if cls._file_handler:
            cls._file_handler.setLevel(cls._log_level)

        for logger in cls._loggers.values():
            logger.setLevel(cls._log_level)

    @classmethod
    def set_external_handler(cls, handler: Callable[[int, str, Optional[Dict[str, Any]]], None]):
        """
        Register external log handler (e.g., for Sentry, Firebase, etc.).

        Parameters
        ----------
        handler : Callable
            Function that takes (level, message, metadata) and sends to external service

        Examples
        --------
        >>> def sentry_handler(level, message, metadata):
        ...     if level >= logging.ERROR:
        ...         sentry_sdk.capture_message(message, extra=metadata)
        >>> MeVeLogger.set_external_handler(sentry_handler)
        """
        cls._external_handler = handler

    @classmethod
    def _add_to_buffer(cls, level: int, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add log entry to buffer for error reporting."""
        cls._log_buffer.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": logging.getLevelName(level),
                "message": message,
                "metadata": metadata or {},
            }
        )

        # Keep buffer size manageable
        if len(cls._log_buffer) > cls._buffer_size:
            cls._log_buffer.pop(0)

        # Call external handler if registered
        if cls._external_handler:
            try:
                cls._external_handler(level, message, metadata)
            except Exception as e:
                # Don't let external handler errors break logging
                print(f"External log handler failed: {e}", file=sys.stderr)

    @classmethod
    def get_log_buffer(cls) -> List[Dict[str, Any]]:
        """Get copy of the log buffer (useful for error reporting)."""
        return cls._log_buffer.copy()

    @classmethod
    def clear_log_buffer(cls):
        """Clear the log buffer."""
        cls._log_buffer.clear()


# Convenience function for getting a logger
def get_logger(name: str) -> Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    Logger
        Enhanced logger instance

    Examples
    --------
    >>> from meve.utils import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    >>> logger.debug("Detailed debug information", {"user_id": 123})
    >>> logger.warning("Something unexpected happened")
    >>> logger.error("An error occurred", error=exception)
    >>> logger.success("Operation completed successfully")
    >>>
    >>> # Performance timing
    >>> logger.time("data-fetch")
    >>> # ... do work ...
    >>> logger.timeEnd("data-fetch")
    >>>
    >>> # Scoped logger
    >>> auth_logger = logger.scope("Auth")
    >>> auth_logger.info("User login attempt")
    >>>
    >>> # Child logger with context
    >>> user_logger = logger.child({"user_id": 123, "session": "abc"})
    >>> user_logger.info("Action performed")  # Includes context automatically
    """
    return MeVeLogger.get_logger(name)


# Create a default logger instance for convenience
logger = get_logger("meve")


# Convenience object for quick access (similar to React Native example)
log = {
    "debug": lambda msg, meta=None: logger.debug(msg, meta),
    "info": lambda msg, meta=None: logger.info(msg, meta),
    "warn": lambda msg, meta=None: logger.warn(msg, meta),
    "error": lambda msg, err=None, meta=None: logger.error(msg, err, meta),
    "success": lambda msg, meta=None: logger.success(msg, meta),
    "time": lambda label: logger.time(label),
    "timeEnd": lambda label: logger.timeEnd(label),
    "divider": lambda label=None: logger.divider(label),
    "lifecycle": lambda event, meta=None: logger.lifecycle(event, meta),
    "network": lambda method, url, meta=None: logger.network(method, url, meta),
    "navigation": lambda loc, params=None: logger.navigation(loc, params),
    "scope": lambda prefix: logger.scope(prefix),
    "child": lambda ctx: logger.child(ctx),
    "group": lambda label: logger.group(label),
    "groupEnd": lambda label="": logger.groupEnd(label),
    "table": lambda data: logger.table(data),
}


# Auto-configure on import if config exists
def _auto_configure():
    """Auto-configure logging from default config if available."""
    project_root = Path(__file__).parent.parent.parent
    dev_config = project_root / "config" / "development.yaml"

    if dev_config.exists():
        MeVeLogger.configure_from_yaml(str(dev_config))
    else:
        # Fallback to basic configuration
        MeVeLogger.configure(level="INFO")


# Run auto-configuration on module import
_auto_configure()

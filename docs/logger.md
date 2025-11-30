# MeVe Logger - Production-Level Logging

## Quick Start

The MeVe logger is globally available throughout your application:

```python
# Import the global logger
from meve import logger, log, get_logger

# Basic usage
logger.info("Processing started")
logger.debug("Debug information", {"key": "value"})
logger.success("Task completed!")
logger.warn("Warning message")
logger.error("Error occurred", error=exception)

# Convenience object (similar to React Native)
log["info"]("Quick message")
log["success"]("Done!")
log["time"]("operation")
# ... do work ...
log["timeEnd"]("operation")

# Module-specific logger
my_logger = get_logger(__name__)
my_logger.info("Module action")
```

## Features

✅ **Color-coded console output** - Beautiful, easy-to-read logs  
✅ **Multiple log levels** - DEBUG, INFO, WARNING, ERROR, CRITICAL  
✅ **Structured metadata** - Attach JSON metadata to any log  
✅ **Performance timing** - Built-in timing utilities  
✅ **Error tracking** - Automatic stack traces for errors  
✅ **Scoped loggers** - Create loggers with prefixes  
✅ **Child loggers** - Persistent context across logs  
✅ **Log buffering** - Keep recent logs for error reporting  
✅ **External integration** - Easy hooks for Sentry, etc.  
✅ **Production-safe** - Respects environment configuration  

## Usage Patterns

### 1. Default Global Logger

```python
from meve import logger

logger.info("Starting process")
logger.debug("Details here", {"user_id": 123})
logger.success("Process complete!")
```

### 2. Convenience Log Object

```python
from meve import log

log["info"]("Quick message")
log["error"]("Something failed", error)
log["time"]("fetch-data")
# ... work ...
log["timeEnd"]("fetch-data")
```

### 3. Module-Specific Loggers

```python
from meve import get_logger

logger = get_logger(__name__)  # Best practice
logger.info("Module initialized")
```

### 4. Scoped Loggers

```python
from meve import logger

auth_logger = logger.scope("Auth")
db_logger = logger.scope("Database")

auth_logger.info("Login attempt")  # [Auth] Login attempt
db_logger.info("Query executed")   # [Database] Query executed
```

### 5. Child Loggers with Context

```python
from meve import logger

# Context persists across all logs
user_logger = logger.child({
    "user_id": "123",
    "session": "abc"
})

user_logger.info("Action 1")  # Includes user_id and session
user_logger.info("Action 2")  # Includes user_id and session
```

### 6. Performance Timing

```python
from meve import logger

logger.time("operation")
# ... do work ...
duration = logger.timeEnd("operation")  # Logs duration in ms
```

### 7. Network Logging

```python
logger.network("GET", "/api/users", {"status": 200})
logger.network("POST", "/api/login", {"response_time": "45ms"})
```

### 8. Lifecycle Events

```python
logger.lifecycle("App Started", {"version": "0.2.0"})
logger.lifecycle("Database Connected")
```

### 9. Error Handling

```python
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", error=e, metadata={
        "operation": "risky_operation",
        "params": params
    })
```

### 10. Grouped Logs

```python
logger.group("User Registration")
logger.info("Validating email")
logger.info("Creating account")
logger.success("Registration complete")
logger.groupEnd("Registration")
```

## Configuration

### From YAML (Automatic)

The logger auto-configures from `config/development.yaml`:

```yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/meve_dev.log"
  use_colors: true
  enable_timestamps: true
  enable_stack_trace: true
```

### Programmatic Configuration

```python
from meve import MeVeLogger

MeVeLogger.configure(
    level="DEBUG",
    log_file="logs/app.log",
    use_colors=True,
    enable_stack_trace=True
)
```

### Dynamic Level Changes

```python
from meve import MeVeLogger

# Change log level at runtime
MeVeLogger.set_level("DEBUG")
MeVeLogger.set_level("INFO")
```

## External Service Integration

Integrate with Sentry, Firebase, or any logging service:

```python
from meve import MeVeLogger
import sentry_sdk

def sentry_handler(level, message, metadata):
    if level >= logging.ERROR:
        sentry_sdk.capture_message(message, extras=metadata)

MeVeLogger.set_external_handler(sentry_handler)
```

## Log Buffer for Error Reporting

```python
from meve import MeVeLogger

# Get recent logs (useful for error reports)
recent_logs = MeVeLogger.get_log_buffer()

# Clear buffer
MeVeLogger.clear_log_buffer()
```

## Examples

Run the comprehensive examples:

```bash
python3 examples/global_logger_demo.py
```

## Color Output

The logger uses ANSI colors for beautiful console output:

- 🔍 **DEBUG** - Gray
- ℹ️  **INFO** - Blue  
- ⚠️  **WARNING** - Yellow
- ❌ **ERROR** - Red
- 🔥 **CRITICAL** - Red background

## Best Practices

1. **Use `__name__` for module loggers**: `get_logger(__name__)`
2. **Add metadata for context**: `logger.info("msg", {"key": "value"})`
3. **Use scoped loggers for subsystems**: `logger.scope("API")`
4. **Use child loggers for persistent context**: `logger.child({"user": 123})`
5. **Time performance-critical operations**: `logger.time()` / `logger.timeEnd()`
6. **Always pass exceptions to error logs**: `logger.error("msg", error=e)`

## API Reference

### Logger Methods

- `debug(message, metadata=None)` - Debug level log
- `info(message, metadata=None)` - Info level log
- `warn(message, metadata=None)` - Warning level log
- `error(message, error=None, metadata=None)` - Error level log
- `critical(message, error=None, metadata=None)` - Critical level log
- `success(message, metadata=None)` - Success log (info level)
- `time(label)` - Start performance timer
- `timeEnd(label)` - End timer and log duration
- `lifecycle(event, metadata=None)` - Log lifecycle event
- `network(method, url, metadata=None)` - Log network request
- `navigation(location, params=None)` - Log navigation
- `divider(label=None)` - Print visual divider
- `group(label)` - Start log group
- `groupEnd(label="")` - End log group
- `table(data)` - Display data as table
- `scope(prefix)` - Create scoped logger
- `child(context)` - Create child logger with context

### MeVeLogger Class Methods

- `configure(**kwargs)` - Configure logger settings
- `configure_from_yaml(path)` - Load config from YAML
- `get_logger(name)` - Get/create logger instance
- `set_level(level)` - Change log level
- `set_external_handler(handler)` - Register external handler
- `get_log_buffer()` - Get recent logs
- `clear_log_buffer()` - Clear log buffer

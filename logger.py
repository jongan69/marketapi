import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from pythonjsonlogger.jsonlogger import JsonFormatter

# Configure the JSON logger
class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def setup_logger(name: str = "market_api") -> logging.Logger:
    """Setup and configure the logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create JSON formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(module)s %(function)s %(line)s %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)
    return logger

# Create the logger instance
logger = setup_logger()

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "request_id": request.headers.get("X-Request-ID"),
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
            }
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request.headers.get("X-Request-ID"),
                    "status_code": response.status_code,
                    "processing_time": f"{process_time:.4f}s",
                }
            )
            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request.headers.get("X-Request-ID"),
                    "error": str(e),
                    "processing_time": f"{process_time:.4f}s",
                }
            )
            raise

def log_endpoint_access(endpoint: str, **kwargs) -> None:
    """Log endpoint access with additional context"""
    logger.info(
        f"Endpoint accessed: {endpoint}",
        extra={
            "endpoint": endpoint,
            **kwargs
        }
    )

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log errors with context"""
    logger.error(
        str(error),
        extra={
            "error_type": type(error).__name__,
            "context": context or {}
        }
    )

def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics"""
    logger.info(
        f"Performance metric: {operation}",
        extra={
            "operation": operation,
            "duration": f"{duration:.4f}s",
            **kwargs
        }
    ) 
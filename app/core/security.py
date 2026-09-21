"""
Security and Rate Limiting for Polar Navigator AI
Provides optional API token authentication and in-memory request rate limiting.
"""

import os
import time
from collections import defaultdict
from fastapi import HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-Polar-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

EXPECTED_API_KEY = os.getenv("POLAR_API_KEY", "")


class RateLimiter:
    """Sliding-window request rate limiter per client IP."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0, requests_per_minute: int = None):
        if requests_per_minute is not None:
            self.max_requests = requests_per_minute
            self.window_seconds = 60.0
        else:
            self.max_requests = max_requests
            self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        # Filter timestamps within window
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "150")))


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verifies API key if POLAR_API_KEY is configured in the environment."""
    if not EXPECTED_API_KEY:
        # Auth is optional in demo/evaluation mode
        return True
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Polar-API-Key header",
        )
    return True


def check_rate_limit(request: Request):
    """Dependency to check client request rate limit."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum requests per minute reached.",
        )
    return True


verify_api_key_header = verify_api_key

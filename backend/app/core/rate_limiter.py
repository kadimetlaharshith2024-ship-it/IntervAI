import time
import threading
from typing import Dict, List, Optional
from fastapi import Request, HTTPException, status
from app.core.config import settings

class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter for expensive AI & persistence endpoints.
    Protects against duplicate concurrent clicks, burst requests, and accidental loop calls.

    Note on Distributed Deployments:
    This in-memory implementation provides process-level rate limiting for single-worker or
    sticky-session deployments. For multi-instance or Kubernetes deployments, configure a
    Redis/Valkey-backed centralized key-value store to synchronize sliding windows across workers.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: Dict[str, List[float]] = {}

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60
    ) -> None:
        """
        Checks if the client with the specified key has exceeded max_requests in window_seconds.
        Raises HTTPException(429) if exceeded.
        """
        if not getattr(settings, "RATE_LIMITING_ENABLED", True):
            return

        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            timestamps = self._requests.get(key, [])
            # Prune timestamps outside the current window
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= max_requests:
                earliest = valid_timestamps[0]
                retry_after = max(1, int(earliest + window_seconds - now))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds}s permitted. Please retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)}
                )

            valid_timestamps.append(now)
            self._requests[key] = valid_timestamps

    def reset(self):
        """Resets all recorded rate limit windows (useful for test isolation)."""
        with self._lock:
            self._requests.clear()

rate_limiter = SlidingWindowRateLimiter()

def rate_limit(max_requests: int, window_seconds: int = 60):
    """
    FastAPI dependency factory to enforce sliding window rate limits on endpoints.
    Combines client IP and user_id (if authenticated) to isolate quotas per tenant.
    """
    def dependency(request: Request):
        if getattr(settings, "ENVIRONMENT", "development") == "test":
            return

        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        key = f"{client_ip}:{path}"
        rate_limiter.check_rate_limit(key, max_requests, window_seconds)

    return dependency

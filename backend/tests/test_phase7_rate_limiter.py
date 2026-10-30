import pytest
from fastapi import HTTPException
from app.core.rate_limiter import SlidingWindowRateLimiter

def test_rate_limiter_allows_under_quota():
    limiter = SlidingWindowRateLimiter()
    # 3 requests allowed in 60s
    for _ in range(3):
        limiter.check_rate_limit("client_1:next-question", max_requests=3, window_seconds=60)

def test_rate_limiter_blocks_over_quota():
    limiter = SlidingWindowRateLimiter()
    for _ in range(3):
        limiter.check_rate_limit("client_2:next-question", max_requests=3, window_seconds=60)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit("client_2:next-question", max_requests=3, window_seconds=60)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers

def test_rate_limiter_isolates_different_keys():
    limiter = SlidingWindowRateLimiter()
    for _ in range(3):
        limiter.check_rate_limit("client_a:endpoint", max_requests=3, window_seconds=60)

    # client_b should still be allowed
    limiter.check_rate_limit("client_b:endpoint", max_requests=3, window_seconds=60)

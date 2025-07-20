import asyncio
import time
import pytest

from nexusvoice.utils.debug import AsyncRateLimiter


@pytest.mark.asyncio
async def test_async_rate_limiter_set_get_instance():
    """Test that AsyncRateLimiter can set and get instances correctly."""
    # Create a rate limiter instance
    rate_limiter1 = AsyncRateLimiter(rate=10, per_seconds=1)
    rate_limiter2 = AsyncRateLimiter(rate=5, per_seconds=1)
    
    # Test default instance
    AsyncRateLimiter.set_instance(rate_limiter1)
    assert AsyncRateLimiter.get_instance() is rate_limiter1
    
    # Test named instances
    AsyncRateLimiter.set_instance(rate_limiter1, "instance1")
    AsyncRateLimiter.set_instance(rate_limiter2, "instance2")
    
    assert AsyncRateLimiter.get_instance("instance1") is rate_limiter1
    assert AsyncRateLimiter.get_instance("instance2") is rate_limiter2
    
    # Test non-existent instance
    assert AsyncRateLimiter.get_instance("non_existent") is None


@pytest.mark.asyncio
async def test_async_rate_limiter_acquire():
    """Test that AsyncRateLimiter.acquire limits the rate of operations."""
    # Create a rate limiter with 5 tokens per second
    rate = 5
    per_seconds = 1
    rate_limiter = AsyncRateLimiter(rate=rate, per_seconds=per_seconds)
    
    # Should be able to acquire immediately for the first 'rate' times
    success_count = 0
    start_time = time.monotonic()
    
    # Try to acquire more tokens than the rate allows
    for _ in range(int(rate * 2)):
        result = await rate_limiter.acquire()
        if result:
            success_count += 1
    print(f"Success count: {success_count}")
    # The first batch should succeed immediately
    # We may get more than 'rate' due to token refill timing, but we should get at least 'rate'
    assert success_count >= rate
    
    # Now check rate limiting over a longer period (2 seconds)
    acquisition_duration = 2.0  # seconds
    acquisition_start = time.monotonic()
    acquisition_end = acquisition_start + acquisition_duration
    
    # Track additional successful acquisitions
    additional_success = 0
    
    # Keep trying to acquire until our time is up
    while time.monotonic() < acquisition_end:
        result = await rate_limiter.acquire()
        if result:
            additional_success += 1
        # Small delay to prevent tight loop
        await asyncio.sleep(0.01)
    
    # Calculate the theoretical maximum tokens that should be available
    # over the acquisition period (excluding initial tokens which were already consumed)
    max_tokens_per_second = rate / per_seconds
    theoretical_max = max_tokens_per_second * acquisition_duration
    
    print(f"Additional success: {additional_success}")
    print(f"Theoretical max: {theoretical_max}")
    
    # Assert that we didn't acquire more tokens than theoretically possible
    # We add a small buffer (0.5) to account for timing variations
    assert additional_success <= (theoretical_max + 0.5), f"Acquired {additional_success} tokens, expected max {theoretical_max}"
    
    # We should also have acquired some tokens during this period
    assert additional_success > 0, "No tokens were acquired during the test period"


@pytest.mark.asyncio
async def test_async_rate_limiter_concurrent():
    """Test AsyncRateLimiter's behavior with concurrent requests."""
    # Create a rate limiter with 3 tokens per second
    rate = 3
    rate_limiter = AsyncRateLimiter(rate=rate, per_seconds=1)
    
    # Create several concurrent tasks trying to acquire tokens
    async def worker():
        return await rate_limiter.acquire()
    
    # Run 5 workers concurrently (more than the rate limit)
    results = await asyncio.gather(*[worker() for _ in range(int(rate * 2))])
    
    # Expected behavior is that at least 3 should succeed, but due to timing,
    # we might get more than exactly 3. What's important is that not all can succeed.
    print(f"Results: {results}")
    assert results.count(True) >= rate
    assert results.count(False) >= 1


@pytest.mark.asyncio
async def test_async_rate_limiter_concurrent_multi():
    """Test that multiple AsyncRateLimiter instances can operate concurrently and independently."""
    # Create two rate limiters with different rates
    rate_limiter1 = AsyncRateLimiter(rate=3, per_seconds=1)
    rate_limiter2 = AsyncRateLimiter(rate=5, per_seconds=1)
    
    # Create worker functions for each rate limiter
    async def worker1():
        return await rate_limiter1.acquire()
    
    async def worker2():
        return await rate_limiter2.acquire()
    
    # Run workers for both rate limiters concurrently
    # We'll run more workers than the allowed rate for each limiter
    results1 = await asyncio.gather(*[worker1() for _ in range(6)])  # 6 > rate of 3
    results2 = await asyncio.gather(*[worker2() for _ in range(8)])  # 8 > rate of 5
    
    # Print the results for debugging
    print(f"Rate limiter 1 results: {results1}")
    print(f"Rate limiter 2 results: {results2}")
    
    # Verify that each rate limiter enforced its own limit
    assert results1.count(True) >= 3, "Rate limiter 1 didn't grant enough tokens"
    assert results1.count(False) >= 1, "Rate limiter 1 didn't enforce its limit"
    
    assert results2.count(True) >= 5, "Rate limiter 2 didn't grant enough tokens"
    assert results2.count(False) >= 1, "Rate limiter 2 didn't enforce its limit"
    
    # Now test true concurrency between the two limiters
    # by interleaving tasks from both limiters
    mixed_tasks = []
    for i in range(10):  # Create 10 tasks, alternating between limiters
        if i % 2 == 0:
            mixed_tasks.append(worker1())
        else:
            mixed_tasks.append(worker2())
            
    mixed_results = await asyncio.gather(*mixed_tasks)
    print(f"Mixed limiter results: {mixed_results}")
    
    # Count successes for each limiter (even indices for limiter1, odd for limiter2)
    limiter1_successes = sum(1 for i in range(0, len(mixed_results), 2) if mixed_results[i] is True)
    limiter2_successes = sum(1 for i in range(1, len(mixed_results), 2) if mixed_results[i] is True)
    
    # Verify that each limiter worked independently
    # We're giving some leeway here as the exact timing can affect results
    assert 1 <= limiter1_successes <= 3, f"Rate limiter 1 gave unexpected number of successes: {limiter1_successes}"
    assert 1 <= limiter2_successes <= 5, f"Rate limiter 2 gave unexpected number of successes: {limiter2_successes}"

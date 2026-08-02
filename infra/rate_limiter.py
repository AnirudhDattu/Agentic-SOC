# infra/rate_limiter.py
import time
import threading


class RateLimiter:
    """Daily-quota + minimum-interval gate for a metered API.

    allow() is safe to call from a single-threaded loop (main.py's while-True)
    and cheap enough to check on every event.
    """

    def __init__(self, max_per_day: int, min_interval: float):
        self._lock = threading.Lock()
        self._day = None
        self._count = 0
        self._max_per_day = max_per_day
        self._min_interval = min_interval
        self._last_call = 0.0

    def allow(self) -> bool:
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if today != self._day:
                self._day = today
                self._count = 0

            if self._count >= self._max_per_day:
                return False
            if time.time() - self._last_call < self._min_interval:
                return False

            self._count += 1
            self._last_call = time.time()
            return True

    @property
    def remaining_today(self) -> int:
        with self._lock:
            return max(0, self._max_per_day - self._count)

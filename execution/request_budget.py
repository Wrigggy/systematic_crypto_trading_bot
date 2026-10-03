"""Shared rolling request ceiling; urgent work uses reserved capacity, not bypasses."""

import time
from collections import deque


class RequestBudgetExceeded(RuntimeError):
    """No request was sent; retain the live order and defer the work."""


class RequestBudget:
    def __init__(self, limit=5, reserve=1, clock=time.monotonic):
        if (
            not isinstance(limit, int)
            or not isinstance(reserve, int)
            or not 0 <= reserve < limit
        ):
            raise ValueError("Require integer 0 <= emergency reserve < request limit")
        self.limit, self.reserve, self.clock = limit, reserve, clock
        self.sent = deque()
        self.total = 0

    def take(self, urgent=False):
        now = self.clock()
        while self.sent and now - self.sent[0] >= 60:
            self.sent.popleft()
        cap = self.limit if urgent else self.limit - self.reserve
        if len(self.sent) >= cap:
            raise RequestBudgetExceeded("Shared request budget exhausted")
        self.sent.append(now)
        self.total += 1

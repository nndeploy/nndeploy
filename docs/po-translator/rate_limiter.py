import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls, period):
        """
        初始化 RateLimiter
        :param max_calls: 在指定时间段内允许的最大调用次数
        :param period: 时间段的长度（秒）
        """
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill_time = time.time()
        self.call_times = deque()

    def _refill_tokens(self):
        """
        重新填充令牌
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_refill_time
        self.last_refill_time = current_time

        # 计算应该添加的令牌数量
        new_tokens = elapsed_time / self.period * self.max_calls
        self.tokens = min(self.max_calls, self.tokens + new_tokens)

    def _remove_expired_calls(self):
        """
        移除过期的调用记录
        """
        current_time = time.time()
        while self.call_times and self.call_times[0] < current_time - self.period:
            self.call_times.popleft()

    def __call__(self, func):
        """
        装饰器：限制函数调用频率
        """

        def wrapper(*args, **kwargs):
            current_time = time.time()
            self._remove_expired_calls()

            # 检查当前时间段内的调用次数是否已达到上限
            if len(self.call_times) >= self.max_calls:
                # 如果达到上限，计算需要等待的时间
                wait_time = self.call_times[0] + self.period - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()  # 更新当前时间

            # 添加调用记录
            self.call_times.append(current_time)
            return func(*args, **kwargs)

        return wrapper


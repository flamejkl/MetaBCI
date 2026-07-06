import time
import numpy as np
from config import VOTER_DECAY, VOTER_LOCK_FRAMES, VOTER_LOCK_DURATION, VOTER_THRESHOLD

class AdvancedVoter:
    def __init__(self, decay=VOTER_DECAY, lock_frames=VOTER_LOCK_FRAMES,
                 lock_duration=VOTER_LOCK_DURATION, threshold=VOTER_THRESHOLD):
        self.decay = decay
        self.lock_frames = lock_frames
        self.lock_duration = lock_duration
        self.threshold = threshold
        self.reset()   # 使用 reset 初始化

    def reset(self):
        """重置投票器状态，每个评测步骤开始时必须调用"""
        self.accumulated = np.zeros(4)
        self.consecutive = 0
        self.locked_dir = None
        self.lock_until = 0.0
        self.last_dir = None

    def update(self, current_prob, timestamp=None):
        """
        current_prob: 长度为4的列表或数组，当前分类器输出的概率
        timestamp: 可选，用于锁定时间判断，默认为time.time()
        返回 (direction_idx, confidence) 或 (None, None) 表示无指令输出
        """
        if timestamp is None:
            timestamp = time.time()
        # 1. 置信度累积（指数衰减）
        self.accumulated = self.decay * self.accumulated + (1 - self.decay) * np.array(current_prob)
        best_idx = np.argmax(self.accumulated)
        best_conf = self.accumulated[best_idx]

        # 2. 锁定检查
        if self.locked_dir is not None and timestamp < self.lock_until:
            # 锁定期间强制输出锁定方向，但不更新累积器（或者保持方向不变）
            return self.locked_dir, best_conf

        # 3. 正常决策
        if best_conf >= self.threshold:
            # 检查是否与上次方向一致
            if self.last_dir == best_idx:
                self.consecutive += 1
            else:
                self.consecutive = 1
            self.last_dir = best_idx

            # 连续达到次数则锁定
            if self.consecutive >= self.lock_frames:
                self.locked_dir = best_idx
                self.lock_until = timestamp + self.lock_duration
                self.consecutive = 0
            return best_idx, best_conf
        else:
            self.consecutive = 0
            self.last_dir = None
            return None, best_conf
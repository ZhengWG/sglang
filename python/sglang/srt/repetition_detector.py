"""Rolling hash repetition loop detection (Rabin-Karp).

Algorithm:
  Every decode step, compute a rolling hash over the last W tokens (the "window").
  The rolling hash uses H = (H_prev * BASE + new_token - old_token * BASE^W) mod MOD,
  so each slide costs O(1) — no need to rehash the full window.

  A dict maps each hash to its occurrence count. When any hash reaches N, we declare
  a loop: N identical W-token windows means the output is cycling deterministically.

  Memory is bounded by a deque(maxlen=W+1) for the token ring buffer, and periodic
  compaction of the hash dict (evict count=1 entries when size > COMPACT_THRESHOLD).
"""

from collections import deque


class RollingHashDetector:
    """Detect deterministic output loops via sliding-window rolling hash.

    Per request, one instance. Memory bounded at O(COMPACT_THRESHOLD).
    """

    # Max hash_counter dict size. GC evicts count=1 entries when exceeded.
    # Repeating hashes (count>=2) survive GC and accumulate toward threshold N.
    # Also caps max detectable repeating pattern length — a pattern producing
    # >COMPACT_THRESHOLD unique hashes per loop will be GC'd before count reaches N.
    # Production worst case: ~400 unique hashes/loop. 10000 = 25x headroom, ~1MB/request.
    COMPACT_THRESHOLD = 10000
    BASE = 131
    MOD = (1 << 61) - 1  # Mersenne prime, collision probability ~4.3e-19

    def __init__(self, window_size: int = 128, max_repeat: int = 10):
        self.W = window_size
        self.N = max_repeat
        self.ring: deque = deque(maxlen=window_size + 1)
        self.count = 0
        self.hash_counter: dict[int, int] = {}
        self.current_hash = 0
        self.base_pow_w = pow(self.BASE, self.W, self.MOD)

    def check(self, new_token_id: int) -> bool:
        """Process one token. Returns True if repetition loop detected."""
        self.ring.append(new_token_id)
        self.count += 1

        if self.count < self.W:
            # Window not full yet, accumulate hash
            self.current_hash = (
                self.current_hash * self.BASE + new_token_id
            ) % self.MOD
            return False

        if self.count == self.W:
            # Window just became full
            self.current_hash = (
                self.current_hash * self.BASE + new_token_id
            ) % self.MOD
        else:
            # Slide: ring[0] is the token that just left the window
            old_token = self.ring[0]
            self.current_hash = (
                self.current_hash * self.BASE
                + new_token_id
                - old_token * self.base_pow_w
            ) % self.MOD

        c = self.hash_counter.get(self.current_hash, 0) + 1
        self.hash_counter[self.current_hash] = c

        if c >= self.N:
            return True

        self._maybe_compact()
        return False

    def _maybe_compact(self):
        """Remove singleton hashes when dict grows too large."""
        if len(self.hash_counter) <= self.COMPACT_THRESHOLD:
            return
        self.hash_counter = {k: v for k, v in self.hash_counter.items() if v >= 2}
        # Extreme edge case: too many hashes with count >= 2
        if len(self.hash_counter) > self.COMPACT_THRESHOLD:
            self.hash_counter.clear()

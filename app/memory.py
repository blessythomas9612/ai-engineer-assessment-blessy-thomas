"""Short-term conversation memory, kept in process memory.

Each session keeps its last `max_turns` question/answer pairs. Sessions idle for
longer than `ttl_s` are dropped, and at most `max_sessions` are kept (the least
recently used are evicted first), so memory use stays bounded.

Corner cut: memory lives in this process only - it is lost on restart and not
shared between server workers. Production would use Redis or a database.
"""
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


class SessionStore:
    def __init__(self, max_turns: int = 5, ttl_s: float = 1800, max_sessions: int = 1000, clock=time.monotonic):
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, tuple[float, deque]] = OrderedDict()
        self._lock = threading.Lock()   # FastAPI runs sync endpoints in a thread pool

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            self._expire()
            entry = self._sessions.get(session_id)
            return list(entry[1]) if entry else []

    def add(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._expire()
            _, turns = self._sessions.pop(session_id, (None, deque(maxlen=self.max_turns)))
            turns.append(Turn(question, answer))
            self._sessions[session_id] = (self._clock(), turns)      # newest at the end
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)                    # evict least recently used

    def _expire(self) -> None:
        cutoff = self._clock() - self.ttl_s
        while self._sessions:
            oldest_id, (last_used, _) = next(iter(self._sessions.items()))
            if last_used >= cutoff:
                break
            del self._sessions[oldest_id]

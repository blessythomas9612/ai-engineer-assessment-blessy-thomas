from app.memory import SessionStore


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_keeps_only_the_last_turns():
    store = SessionStore(max_turns=2)
    for i in range(3):
        store.add("s1", f"q{i}", f"a{i}")
    assert [t.question for t in store.history("s1")] == ["q1", "q2"]


def test_sessions_are_separate():
    store = SessionStore()
    store.add("s1", "about batman", "a")
    assert store.history("s2") == []


def test_idle_sessions_expire():
    clock = Clock()
    store = SessionStore(ttl_s=60, clock=clock)
    store.add("s1", "q", "a")
    clock.now = 61
    assert store.history("s1") == []


def test_least_recently_used_session_is_evicted():
    store = SessionStore(max_sessions=2)
    store.add("old", "q", "a")
    store.add("mid", "q", "a")
    store.add("old", "q2", "a2")     # "old" becomes most recent
    store.add("new", "q", "a")       # evicts "mid"
    assert store.history("mid") == []
    assert len(store.history("old")) == 2

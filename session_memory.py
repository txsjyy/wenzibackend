import time

class SessionMemoryStore:
    def __init__(self):
        self.store = {}
        self.last_access = {}

    def get(self, session_id):
        self.last_access[session_id] = time.time()
        return self.store.get(session_id, {})

    def set(self, session_id, data):
        self.store[session_id] = data
        self.last_access[session_id] = time.time()

    def delete(self, session_id):
        if session_id in self.store:
            del self.store[session_id]
            del self.last_access[session_id]

    def cleanup(self, ttl=3600):
        now = time.time()
        to_delete = [sid for sid, ts in self.last_access.items() if now - ts > ttl]
        for sid in to_delete:
            self.delete(sid)

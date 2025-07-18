import json
import threading
import os

class LocalFileSessionMemoryStore:
    def __init__(self, file_path="session_memory.jsonl"):
        self.file_path = file_path
        self.lock = threading.Lock()
        # Ensure file exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                pass  # create empty file

    def get(self, session_id):
        with self.lock, open(self.file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("session_id") == session_id:
                        return record["data"]
                except Exception:
                    continue
        return {}

    def set(self, session_id, data):
        # Remove old session first (if exists), then append new
        sessions = []
        with self.lock:
            # Load all existing
            with open(self.file_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("session_id") != session_id:
                            sessions.append(record)
                    except Exception:
                        continue
            # Append new session data
            sessions.append({"session_id": session_id, "data": data})
            # Write all back
            with open(self.file_path, "w") as f:
                for s in sessions:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def delete(self, session_id):
        sessions = []
        with self.lock, open(self.file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("session_id") != session_id:
                        sessions.append(record)
                except Exception:
                    continue
        with open(self.file_path, "w") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def cleanup(self):
        # You can implement periodic cleanup if needed, else leave empty
        pass

    def all_sessions(self):
        # Helper: Return all sessions
        with self.lock, open(self.file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    yield record["session_id"], record["data"]
                except Exception:
                    continue

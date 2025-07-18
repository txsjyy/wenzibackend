import os
from dotenv import load_dotenv
import threading

# Load environment variables from .env
load_dotenv()

class DeepSeekKeyManager:
    def __init__(self):
        self.keys = [
            os.getenv("DEEPSEEK_API_KEY_1"),
        ]
        self.keys = [k for k in self.keys if k]
        if not self.keys:
            raise ValueError("No DeepSeek API keys found in environment variables.")
        self.current = 0
        self.lock = threading.Lock()

    def get_key(self):
        with self.lock:
            return self.keys[self.current]

    def rotate(self):
        with self.lock:
            self.current = (self.current + 1) % len(self.keys)
            return self.keys[self.current]

deepseek_key_manager = DeepSeekKeyManager()

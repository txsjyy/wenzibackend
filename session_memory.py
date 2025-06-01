# import time

# class SessionMemoryStore:
#     def __init__(self):
#         self.store = {}
#         self.last_access = {}

#     def get(self, session_id):
#         self.last_access[session_id] = time.time()
#         return self.store.get(session_id, {})

#     def set(self, session_id, data):
#         self.store[session_id] = data
#         self.last_access[session_id] = time.time()

#     def delete(self, session_id):
#         if session_id in self.store:
#             del self.store[session_id]
#             del self.last_access[session_id]

#     def cleanup(self, ttl=3600):
#         now = time.time()
#         to_delete = [sid for sid, ts in self.last_access.items() if now - ts > ttl]
#         for sid in to_delete:
#             self.delete(sid)
import time
import pickle
from pymongo import MongoClient, ASCENDING

class MongoDBSessionMemoryStore:
    def __init__(self, mongo_uri, db_name='sessions', collection='memory'):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection]
        # TTL index: deletes documents an hour after last_access
        self.collection.create_index([('last_access', ASCENDING)], expireAfterSeconds=3600)

    def get(self, session_id):
        doc = self.collection.find_one({'_id': session_id})
        if not doc or 'data' not in doc:
            return {}
        data = doc['data']
        # Unpickle memory
        if 'memory' in data and data['memory']:
            data['memory'] = pickle.loads(data['memory'])
        return data

    def set(self, session_id, data):
        # Pickle memory for storage
        data_to_store = data.copy()
        if 'memory' in data_to_store and data_to_store['memory']:
            data_to_store['memory'] = pickle.dumps(data_to_store['memory'])
        self.collection.update_one(
            {'_id': session_id},
            {'$set': {'data': data_to_store, 'last_access': time.time()}},
            upsert=True
        )

    def delete(self, session_id):
        self.collection.delete_one({'_id': session_id})

    def cleanup(self):
        pass  # Not needed (MongoDB TTL does this)

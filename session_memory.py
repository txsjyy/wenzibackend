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
        pass  # MongoDB TTL does this automatically

    # --- New methods for login/signup ---
    def upsert_profile(self, session_id, description):
        """Create or update a user profile tied to session_id"""
        self.collection.update_one(
            {"_id": session_id},
            {"$set": {
                "data.description": description,
                "last_access": time.time()
            }},
            upsert=True
        )
        return self.get_profile(session_id)

    def get_profile(self, session_id):
        doc = self.collection.find_one({"_id": session_id})
        if not doc:
            return None
        return {
            "session_id": session_id,
            "description": doc.get("data", {}).get("description", "")
        }

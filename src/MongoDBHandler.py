from pymongo import MongoClient
import pprint

class MongoDBHandler():
    
    def __init__(self, mongo_uri, db_name, timeout = 5000):
        
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout)
            self.db = self.client.get_database(db_name)
            self.client.get_database(db_name).collection_names(include_system_collections=False)
        except Exception as e:
            print('Connection error: Timeout occurred, please check the mongo uri')
            raise
        
    def check_if_collection_exists(self, collection_name):
        
        collections = self.list_collections()
        if(collection_name in collections):
            return True
        
        return False

    def list_collections(self):
        
        collections = self.db.collection_names(include_system_collections=False)
        if(len(collections) == 0):
            print('WARNING: the selected db is empty')
        
        return collections

    def get_collection(self, collection_name, filter = {}):
        
        if(self.check_if_collection_exists):
            results = []
            for document in self.db[collection_name].find(filter):
                results.append(document)
            return results
        else:
            return "Collection does not exist"
        
    def count_documents(self, collection_name, filter = {}):
        
        if(self.check_if_collection_exists):
            return self.db[collection_name].count(filter)
        else:
            return "Collection does not exist"
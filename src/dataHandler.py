class dataHandler():
    
    def __init__(self, MongoDBHandler):
        
        self.MongoDBHandler = MongoDBHandler
    
    def get_db_statistics(self):
        
        stats = {}
        for collection in self.MongoDBHandler.list_collections():
            stats[collection] = {}
            stats[collection]["number_of_docs"] = self.MongoDBHandler.count_documents(collection)
            
        return stats
    
    def get_users(self, filter = {}):
        
        return self.MongoDBHandler.get_collection("users", filter)
        
    def get_gestures(self, filter = {}):
        
        return self.MongoDBHandler.get_collection("gestures", filter)
    
    def get_devices(self, filter = {}):
        
        return self.MongoDBHandler.get_collection("devices", filter)
    
    def get_user_from_device(self, device_id):
        
        device = self.get_devices({"device_id": device_id})
        if(len(device) > 0):
            user = self.get_users({"_id": device[0]["user_id"]})
            return user
        
        return "Device not found"
    
    def get_gestures_from_device(self, device_id):
        
        return self.get_gestures({"device_id": device_id})
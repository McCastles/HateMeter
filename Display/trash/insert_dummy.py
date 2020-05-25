import random
import datetime

from pymongo import MongoClient
client = MongoClient()

DB_NAME = 'dummy'
db = client[DB_NAME]

collections = ['A', 'B', 'C']

for colname in collections:
    collection = db[colname]

    for i in range(random.randint(1, 5)):
        post = {
            "ala": "makota",
            "randnum": random.randint(9999),
            "date": datetime.datetime.utcnow()
        }
        post_id = collection.insert_one(post).inserted_id
        print('Inserting...', post_id)



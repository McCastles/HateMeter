import random
import datetime
from pprint import pprint
import time

from pymongo import MongoClient



def gen_random_post(i):
    return {
        "user_id": i,
        "ala": "makota",
        "randnum": random.randint(1, 9999),
        "date": datetime.datetime.utcnow()
    }

def load_batch(cursor, n):

    # collection_size = collection.count_documents()
    cursor = collection.find().skip(n)


DB_NAME = 'dummy'
WORKER1_IP = '10.0.2.6'

client = MongoClient(WORKER1_IP)
db = client[DB_NAME]

if 'AAA' in db.list_collection_names():
    db['AAA'].drop()

collection = db['AAA']

train = 5

print('\n======= WRITING =======')
for i in range(1, train+1):

    post_id = collection.insert_one( gen_random_post(i) ).inserted_id
    print('\nInserting...', post_id)


shown = 0

print('\n======= COUNTING =======')
print('Cursor points to documents:', collection.count_documents({}))

print('\n======= READING (and adding new) =======')

# test = 3
# i = 0
# for i in range(1, train + test + 1):

latest_doc = 'nothing'
cursor = collection.find()

b = True

while True:

    try:
        doc = cursor.next()
    except StopIteration:
        pprint(latest_doc)
        print('No more new posts! Looking for more...', '\n')
        # load_batch(cursor, shown)
        cursor = collection.find().skip(shown)

    else:
        pprint(doc)
        print('\n')
        shown += 1
        latest_doc = doc

    finally:
        time.sleep(1)

        if b:
            post_id = collection.insert_one( gen_random_post(999) ).inserted_id
            print('\nInserting...', post_id)            
            b = False


'''
for doc in cursor:

    i += 1

    print('\nReading...')
    pprint(collection.find_one( {'user_id':i} ))
    
    if i < test + 1:
        post_id = collection.insert_one( gen_random_post(train+i) ).inserted_id
        print('Inserting...', post_id)
'''


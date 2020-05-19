import datetime
import random
import time
from pprint import pprint
from pymongo import MongoClient

DELAY_SEC = 2
DB_NAME = 'tweet_sentiment'
COMP_NAME = 'DHL'
WORKER1_IP = '10.0.2.6'

client = MongoClient(WORKER1_IP)
db = client[DB_NAME]

companies = db.list_collection_names()
print('Companies:', companies)
cursors = {
    comp_name: db[comp_name].find()
    for comp_name in companies
}

collection = db[COMP_NAME]

shown = 0

print('\n======= COUNTING =======')
print('Cursor points to documents:', collection.count_documents({}))

print('\n======= READING (and adding new) =======')

latest_doc = 'nothing'
cursor = cursors[COMP_NAME]



# while True:
def get_next_post(comp_name):
    # print('\n Post shown already:', shown1)
    print('\nGetting next post...')
    shown1 = shown
    cursor = cursors[comp_name]
    try:
        doc = cursor.next()
    except StopIteration:
        print('No more new posts! Looking for more...', '\n')
        # load_batch(cursor, shown)
        cursor = collection.find().skip(shown)
        pprint(latest_doc)
        return latest_doc

    else:
        print('\n')
        pprint(doc)
        shown1 += 1
        latest_doc = doc
        return doc

    # finally:
        # return doc
        # time.sleep(1)





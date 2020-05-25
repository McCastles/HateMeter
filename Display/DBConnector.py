import datetime
import random
import time
from pprint import pprint
from pymongo import MongoClient

# DB_NAME = 'tweet_sentiment'
DB_NAME = 'histogram_test'
WORKER1_IP = '10.0.2.6'

client = MongoClient(WORKER1_IP)
db = client[DB_NAME]

comp_names = db.list_collection_names()
print('Names of companies:', comp_names)


# Later on we'll change this to choose company
# by choosing a button
'''
collection = db[COMP_NAME]
print('\n======= COUNTING =======')
print('Cursor points to documents:', collection.count_documents({}))
print('\n======= READING (and adding new) =======')
cursor = cursors[COMP_NAME]
'''


# HISTORY
# Reads collection from the beginning
# One tweet at a time
# Use this in historical data display

# Fire update_history_meta callback 
# every HISTORY_PERIOD_SEC seconds
HISTORY_PERIOD_SEC = 2

# Cursors point to the beginning
history_cursors = {
    comp_name: db[comp_name].find()
    for comp_name in comp_names
}

# Counters show how many tweets already shown
# from this company
history_counters = {
    comp_name: 0
    for comp_name in comp_names
}

def update_history(comp_name, counters, verbose=False):
    shown = counters[comp_name]
    if verbose:
        print('========== Next Historical Post ==========')
        print('Already shown:', shown)
    cursor = history_cursors[comp_name]
    try:
        post = cursor.next()
    except StopIteration:

        if verbose:
            print('No more new posts! Looking for more...', '\n')
        cursor = db[comp_name].find().skip(shown)
        
    finally:
        post = cursor.next()
        counters[comp_name] = shown + 1
        if verbose:
            print('\n')
            pprint(post)
            print('\n')
        return post




# HISTOGRAM

# Fire update_histogram_meta callback 
# every HISTOGRAM_PERIOD_SEC seconds
HISTOGRAM_PERIOD_SEC = 20

# For every company, 
# how many tweets there were in a collection
# before the callback fired
histogram_counters = {
    comp_name: db[comp_name].count_documents({})
    for comp_name in comp_names
}

def update_histogram(histogram_counters, verbose=True):
    if verbose:
        print('========== Latest Tweets ==========')
        print('Fetched latest tweets from mongo:')
    distribution = {}

    for comp_name, counter in histogram_counters.items():
        
        collection = db[comp_name]
        actual_size = collection.count_documents({})
        distribution[comp_name] = actual_size - counter        
        if verbose:
            print(
                comp_name, 'has',
                actual_size, 'tweets;',
                'fetched:', actual_size - counter)
    if verbose:
        print('\n')

    return distribution

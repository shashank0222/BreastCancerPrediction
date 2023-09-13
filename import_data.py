from pymongo.mongo_client import MongoClient
import pandas as pd
import json



uri = "mongodb+srv://shashank:shashank@cluster0.ylucjb4.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name

DATABASE_NAME = 'MLProjects'

COLLECTION_NAME =   "breastCancer"

# read the data as dataframe

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data , columns = dataset.feature_names)
df['target'] = dataset.target

# convert the data into json
json_record = list(json.loads(df.T.to_json()).values())

# now dump the data into database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

import pymongo
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv()

mongo_url = os.environ["MONGODB_URL"]

client = pymongo.MongoClient(mongo_url)
print(mongo_url)
db = client.tcr
collection = db.loan
# print(type(collection))

def generate_embedding_local(text:str) -> list[float]:
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    return embeddings.tolist()


def embed_data():
    count = 1
    print('embed_data()')
    for doc in collection.find({}):
    # for doc in collection.find({'embedding':{"$exists":False}}):
        print('In Loop')
        try:
            print(f"[{count}] Embedding({doc['_id']}): {doc['text']}")
            count = count+1
            embedding = generate_embedding_local(doc['text'])
            doc['embedding'] = embedding
            print(doc['embedding'])
            collection.update_one({'_id':doc['_id']}, {'$set': {"embedding":embedding}})
        except:
            print("An exception occured.")
    print('finished')



embed_data()

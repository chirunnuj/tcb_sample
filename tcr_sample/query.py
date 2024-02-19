import pymongo
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

_ = load_dotenv()
mongo_url = os.environ["MONGODB_URL"]
client = pymongo.MongoClient(mongo_url)
db = client.tcr
collection = db.loan

def generate_embedding_local(text:str) -> list[float]:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    return embeddings.tolist()


query = "ข้อมูลสินเชื่อ Hire Purchase"

results = collection.aggregate([
    {"$vectorSearch": {
            "index": "LoanSearch",
            "queryVector": generate_embedding_local(query),
            "path": "embedding",
            "numCandidates": 100,
            "limit": 1,
    }}
])

# print(type(results))
for doc in results:
    print(f'Doc Name: {doc["name"]},\nDoc Text: {doc["text"]}\n')

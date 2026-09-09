import os
import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from azure.ai.projects import AIProjectClient
from azure.identity import ClientSecretCredential
from pydantic import BaseModel

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
tenant_id = os.getenv("TENANT_ID")
project_url = os.getenv("PROJECT_URL")
model = os.getenv("MODEL_NAME")
embed_model = os.getenv("EMBED_MODEL")

credentials = ClientSecretCredential(
    client_id=client_id,
    client_secret=client_secret,
    tenant_id=tenant_id
)

project_client = AIProjectClient(
    endpoint=project_url,
    client_id=client_id,
    credential=credentials
)

# define faiss index
dimension = 3072 # size of each vector
index = faiss.IndexFlatL2(dimension) 
chunks = []

openai_client = project_client.get_openai_client()

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str
    
class StoreRequest(BaseModel):
    text: str
    
class SearchRequest(BaseModel):
    query: str
    k: int = 3
    
class RetrievalQuery(BaseModel):
    query: str
    k: int = 3

@app.post("/retrieval/embed")
def embed_text(payload: EmbedRequest): # create embed for input
    response = openai_client.embeddings.create(
        model=embed_model,
        input=payload.text
    )
    return {"embedding": response.data[0].embedding}

@app.post("/retrieval/store")
def store_text(payload: StoreRequest): # store the text and its embedding in the FAISS index
    response = openai_client.embeddings.create(
        model=embed_model,
        input=payload.text
    )
    embedding = np.array(response.data[0].embedding).astype('float32')
    
    index.add(np.array([embedding]))
    chunks.append(payload.text)
    
    return {"data_stored": payload.text}

@app.post("/retrieval/search")
def search(payload: SearchRequest): # search for the k=3 most similar chunks in the FAISS index
    response = openai_client.embeddings.create(
        model=embed_model,
        input=payload.query
    )
    # convert the query text into an embedding vector
    query_embedding = np.array(response.data[0].embedding).astype('float32')
    D, I = index.search(np.array([query_embedding]), payload.k)
    
    results = []
    
    # iterate over the indices of the nearest neighbors returned by faiss search
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
            
    return{
        "query": payload.query,
        "results": results
    }
    
@app.post("/retrieval/query")
def rag_query(payload: RetrievalQuery): # perform a retrieval-augmented generation query
    response = openai_client.embeddings.create(
        model=embed_model,
        input=payload.query
    )
    query_embedding = np.array(response.data[0].embedding).astype('float32')
    
    D, I = index.search(np.array([query_embedding]), payload.k)  # assuming k=3 for retrieval
    
    retrieved_chunks = []
    for idx in I[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            
    context = "\n\n".join(retrieved_chunks)
    
    final_prompt = f"""Context:
Use the following context to answer the query.
{context}

Query:
{payload.query}"""

    response = openai_client.responses.create(
        model=model,
        input=final_prompt
    )
    
    return{
        "query": payload.query,
        "context_used": context,
        "response": response.output_text
    }

@app.get("/")
def read_root():
    return {"msg": "Discord RAG API is running"}

@app.get("/test")
def test():
    response = openai_client.responses.create(
        model=model,
        input="What's the weather like in London today? Give me a 10-sentence summary."
    )
    return {"response": response.output_text}

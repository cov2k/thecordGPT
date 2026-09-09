import os
from dotenv import load_dotenv
from fastapi import FastAPI
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
tenant_id = os.getenv("TENANT_ID")
project_url = os.getenv("PROJECT_URL")
model = os.getenv("MODEL_NAME")

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

openai_client = project_client.get_openai_client()

app = FastAPI()

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

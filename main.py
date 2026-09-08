from fastapi import FastAPI

app = FastAPI() # Create an instance of the FastAPI application

@app.get("/")
def read_root():
    return {"msg": "Discord RAG API is running"}
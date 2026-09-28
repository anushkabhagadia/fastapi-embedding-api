from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

# Load the spaCy model
# Note: You must run 'python -m spacy download en_core_web_md' before running this locally
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    raise RuntimeError("Spacy model 'en_core_web_md' not found. Please download it first.")

app = FastAPI(title="Generative AI Embedding API", version="1.0")

class TextRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    text: str
    embedding: list[float]

@app.post("/embed", response_model=EmbeddingResponse)
def get_embedding(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Process the text using spaCy
    doc = nlp(request.text)
    
    # Get the vector representation (averages word vectors for the whole sentence)
    embedding = doc.vector.tolist()
    
    return {"text": request.text, "embedding": embedding}
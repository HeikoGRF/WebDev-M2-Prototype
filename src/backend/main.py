from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    init_embedding_model()
    init_perplexity_model()
    yield
    # Shutdown: cleanup (if needed)

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app runs on port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models
embedding_model = None
perplexity_model = None
perplexity_tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Request model
class TextRequest(BaseModel):
    text: str

# Initialize the embedding model
def init_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        try:
            embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
            print("Embedding model loaded successfully!")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            raise

# Initialize the perplexity model (SmolLM-135M)
def init_perplexity_model():
    global perplexity_model, perplexity_tokenizer
    if perplexity_model is None:
        print("Loading SmolLM-135M for perplexity calculation...")
        try:
            perplexity_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            perplexity_model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM-135M"
            ).to(device)
            perplexity_model.eval()
            print("SmolLM-135M loaded successfully!")
        except Exception as e:
            print(f"Failed to load perplexity model: {e}")
            raise

def calculate_perplexity(text: str) -> float:
    """Calculate perplexity of text using SmolLM-135M"""
    if perplexity_model is None or perplexity_tokenizer is None:
        init_perplexity_model()
    
    # Tokenize input
    encodings = perplexity_tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    
    # Need at least 2 tokens to calculate perplexity
    if input_ids.size(1) < 2:
        return float('inf')
    
    with torch.no_grad():
        outputs = perplexity_model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


@app.get("/")
def read_root():
    return {
        "message": "Embedding API is running",
        "embedding_model_loaded": embedding_model is not None,
        "perplexity_model_loaded": perplexity_model is not None
    }

@app.post("/embed")
async def generate_embedding(request: TextRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if embedding_model is None:
            init_embedding_model()
        
        if perplexity_model is None:
            init_perplexity_model()
        
        # Generate embedding using sentence transformer
        embedding = embedding_model.encode(request.text)
        embedding_list = embedding.tolist()
        
        # Calculate perplexity using SmolLM-135M
        perplexity = calculate_perplexity(request.text)
        
        return {
            "embedding": embedding_list,
            "perplexity": perplexity
        }
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)

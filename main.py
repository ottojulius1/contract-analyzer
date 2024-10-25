from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contract-analyzer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
async def root():
    return {"message": "Contract Analyzer API is running"}

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    try:
        # Test OpenAI connection first
        try:
            test_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say hello"}
                ]
            )
            print("OpenAI test successful")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"OpenAI API error: {str(e)}"}
            )

        # If we get here, OpenAI is working
        return JSONResponse(content={
            "summary": "Test summary",
            "key_terms": [
                {"term": "Test Term", "value": "Test Value"}
            ],
            "risks": ["Test Risk"]
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error: {str(e)}"}
        )

@app.options("/analyze")
async def analyze_options():
    return JSONResponse(content={})

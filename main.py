from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import openai
import PyPDF2
import io
import os
import json

app = FastAPI()

# Updated CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contract-analyzer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class AnalysisResponse(BaseModel):
    summary: str
    key_terms: List[dict]
    risks: List[str]

@app.get("/")
async def root():
    return {"message": "Contract Analyzer API is running"}

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF files are supported"}
        )
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Convert to text
        try:
            pdf = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Error reading PDF: {str(e)}"}
            )

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "No text could be extracted from the PDF"}
            )

        # Simple test response
        test_response = {
            "summary": "This is a test summary of the contract.",
            "key_terms": [
                {"term": "Duration", "value": "12 months"},
                {"term": "Payment", "value": "$1,000 per month"}
            ],
            "risks": ["This is a test risk"]
        }
        
        return JSONResponse(content=test_response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing request: {str(e)}"}
        )

@app.options("/analyze")
async def analyze_options():
    return JSONResponse(content={})

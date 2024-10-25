from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import PyPDF2
import io
import os
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key not found!")
else:
    logger.info("OpenAI API key configured")
    openai.api_key = api_key

class AnalysisResponse(BaseModel):
    summary: str
    key_terms: List[dict]
    risks: List[str]

def extract_text_from_pdf(file_bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read and extract text
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        logger.info("Making OpenAI API request...")
        
        # Make API request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using 3.5 to reduce costs during testing
            messages=[
                {"role": "system", "content": """You are a legal document analyzer. Analyze the contract and provide:
                    1. A clear summary
                    2. Key terms and their values
                    3. Potential risks or missing elements
                    Format the response as JSON with keys: summary, key_terms (array of {term, value}), risks (array)"""},
                {"role": "user", "content": f"Analyze this contract:\n\n{text[:4000]}"}
            ],
            temperature=0.2
        )
        
        logger.info("Received response from OpenAI")
        
        # Parse the response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return AnalysisResponse(**result)
    
    except openai.error.AuthenticationError as e:
        logger.error(f"OpenAI Authentication Error: {str(e)}")
        raise HTTPException(status_code=500, detail="OpenAI API key is invalid")
    
    except openai.error.APIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error parsing OpenAI response")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the contract accurately and concisely."},
                {"role": "user", "content": f"Contract text: {text[:4000]}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Question answering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

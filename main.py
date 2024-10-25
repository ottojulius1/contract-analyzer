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
openai.api_key = os.getenv("OPENAI_API_KEY")

class AnalysisResponse(BaseModel):
    summary: str
    key_terms: List[dict]
    risks: List[str]

class QuestionResponse(BaseModel):
    answer: str

def extract_text_from_pdf(file_bytes):
    pdf_file = io.BytesIO(file_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_contract_with_ai(text: str) -> AnalysisResponse:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
        
        # Parse the JSON response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        analysis = analyze_contract_with_ai(text)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the contract accurately and concisely."},
                {"role": "user", "content": f"Contract text: {text[:4000]}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        
        return QuestionResponse(answer=response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

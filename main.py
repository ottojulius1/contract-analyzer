from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from typing import Dict, List
import PyPDF2
import io
import os
import json
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contract-analyzer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file_bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def detect_document_type(text: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a legal document classifier. 
                Analyze the document and determine its type. Return ONLY a JSON object without any additional text or formatting.
                Format:
                {
                    "type": "CONTRACT",
                    "confidence": 0.95,
                    "indicators": ["reason 1", "reason 2"]
                }"""},
                {"role": "user", "content": f"Classify this document:\n\n{text[:2000]}"}
            ],
            temperature=0.2
        )
        
        # Get the response content and parse it
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse document type response: {content}")
            return {
                "type": "UNKNOWN",
                "confidence": 0.5,
                "indicators": ["Error parsing classification"]
            }
            
    except Exception as e:
        logger.error(f"Document type detection error: {str(e)}")
        return {
            "type": "UNKNOWN",
            "confidence": 0.5,
            "indicators": ["Error during classification"]
        }

def analyze_document(text: str, doc_type: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze this legal document and return ONLY a JSON object without any additional text or formatting.
                Format:
                {
                    "summary": "brief overview",
                    "key_terms": [
                        {"term": "term name", "value": "description"}
                    ],
                    "dates": [
                        {"event": "event name", "date": "date value"}
                    ],
                    "risks": ["risk 1", "risk 2"],
                    "flags": [
                        {"severity": "HIGH", "issue": "description", "recommendation": "action"}
                    ]
                }"""},
                {"role": "user", "content": f"Analyze this document:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        # Get the response content and parse it
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse analysis response: {content}")
            return {
                "summary": "Error parsing analysis results",
                "key_terms": [],
                "dates": [],
                "risks": ["Analysis parsing error"],
                "flags": [{"severity": "HIGH", "issue": "Analysis Error", "recommendation": "Please try again"}]
            }
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "summary": "Error during analysis",
            "key_terms": [],
            "dates": [],
            "risks": ["Analysis error occurred"],
            "flags": [{"severity": "HIGH", "issue": "Analysis Error", "recommendation": "Please try again"}]
        }

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "No text could be extracted from the PDF"}
            )
        
        # Detect document type
        doc_type_info = detect_document_type(text)
        logger.info(f"Detected document type: {doc_type_info}")
        
        # Perform analysis
        analysis = analyze_document(text, doc_type_info["type"])
        logger.info("Analysis completed successfully")
        
        # Return combined results
        return JSONResponse(content={
            "document_type": doc_type_info,
            "analysis": analysis
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Analysis failed: {str(e)}"}
        )

@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    logger.info(f"Received question: {question}")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the document accurately and concisely."},
                {"role": "user", "content": f"Document text: {text}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Generated answer: {answer}")
        
        return JSONResponse(content={"answer": answer})
        
    except Exception as e:
        logger.error(f"Question error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Question error: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Legal Document Analyzer API is running"}

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

# Define key clause types to analyze
KEY_CLAUSES = {
    "CONFIDENTIALITY": ["confidentiality", "non-disclosure", "private information"],
    "TERMINATION": ["termination", "contract end", "contract cessation"],
    "INDEMNIFICATION": ["indemnify", "indemnification", "hold harmless"],
    "GOVERNING_LAW": ["governing law", "jurisdiction", "applicable law"],
    "PAYMENT_TERMS": ["payment terms", "compensation", "fees"],
    "LIMITATION_LIABILITY": ["limitation of liability", "liability cap", "damages limit"],
    "INTELLECTUAL_PROPERTY": ["intellectual property", "IP rights", "copyright"],
    "FORCE_MAJEURE": ["force majeure", "act of god", "unforeseen circumstances"]
}

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

def analyze_clauses(text: str) -> Dict:
    """Analyze specific clauses in the document."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze the legal document and extract key clauses. 
                For each clause, provide:
                1. The exact text or summary
                2. Whether it's standard or non-standard language
                3. Potential risks or issues
                4. Recommendations for improvement

                Return ONLY a JSON object in this format:
                {
                    "clauses": [
                        {
                            "type": "clause type",
                            "text": "extracted text or summary",
                            "is_standard": true/false,
                            "risks": ["risk 1", "risk 2"],
                            "recommendations": ["recommendation 1", "recommendation 2"],
                            "standard_deviation": "explanation of how it differs from standard language",
                            "importance": "HIGH/MEDIUM/LOW"
                        }
                    ],
                    "missing_clauses": ["important clause type 1", "important clause type 2"],
                    "unusual_provisions": ["unusual provision 1", "unusual provision 2"]
                }"""},
                {"role": "user", "content": f"Analyze these clauses:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Clause analysis error: {str(e)}")
        return {
            "clauses": [],
            "missing_clauses": [],
            "unusual_provisions": []
        }

def detect_document_type(text: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze the document and determine its type. 
                Return ONLY a JSON object in this format:
                {
                    "type": "CONTRACT/EMPLOYMENT/REAL_ESTATE/LEGAL_BRIEF/CORPORATE/PATENT/REGULATORY/FINANCE",
                    "confidence": 0.95,
                    "indicators": ["reason 1", "reason 2"],
                    "industry": "specific industry if applicable",
                    "jurisdiction": "jurisdiction if mentioned"
                }"""},
                {"role": "user", "content": f"Classify this document:\n\n{text[:2000]}"}
            ],
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Document type detection error: {str(e)}")
        return {
            "type": "UNKNOWN",
            "confidence": 0.5,
            "indicators": ["Error during classification"],
            "industry": "Unknown",
            "jurisdiction": "Unknown"
        }

def analyze_document(text: str, doc_type: str) -> Dict:
    try:
        # Get clause analysis first
        clause_analysis = analyze_clauses(text)
        
        # Get general analysis
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze this legal document and provide a detailed analysis. 
                Return ONLY a JSON object in this format:
                {
                    "summary": "comprehensive summary",
                    "key_terms": [
                        {"term": "term name", "value": "description", "category": "FINANCIAL/LEGAL/OPERATIONAL"}
                    ],
                    "dates": [
                        {"event": "event name", "date": "date value", "importance": "HIGH/MEDIUM/LOW"}
                    ],
                    "risks": [
                        {"description": "risk description", "severity": "HIGH/MEDIUM/LOW", "category": "LEGAL/FINANCIAL/OPERATIONAL"}
                    ],
                    "flags": [
                        {"severity": "HIGH/MEDIUM/LOW", "issue": "description", "recommendation": "action"}
                    ],
                    "action_items": [
                        {"description": "action item", "deadline": "deadline if any", "priority": "HIGH/MEDIUM/LOW"}
                    ]
                }"""},
                {"role": "user", "content": f"Analyze this document:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        general_analysis = json.loads(response.choices[0].message.content)
        
        # Combine analyses
        return {
            **general_analysis,
            "clause_analysis": clause_analysis
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "summary": "Error during analysis",
            "key_terms": [],
            "dates": [],
            "risks": ["Analysis error occurred"],
            "flags": [{"severity": "HIGH", "issue": "Analysis Error", "recommendation": "Please try again"}],
            "action_items": [],
            "clause_analysis": {
                "clauses": [],
                "missing_clauses": [],
                "unusual_provisions": []
            }
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
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the document accurately and concisely. If referring to specific clauses, cite them."},
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
    return {"message": "Enhanced Legal Document Analyzer API is running"}

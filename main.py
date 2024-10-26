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
import math
from datetime import datetime

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

def calculate_complexity_score(text: str) -> float:
    """Calculate document complexity score."""
    words = text.split()
    sentences = text.split('.')
    if not sentences[-1].strip():
        sentences = sentences[:-1]
    
    # Average words per sentence
    avg_words = len(words) / max(len(sentences), 1)
    
    # Complexity factors
    long_words = len([w for w in words if len(w) > 6])
    long_word_percentage = long_words / max(len(words), 1)
    
    # Score from 0-100
    score = min((avg_words * 4 + long_word_percentage * 100) / 2, 100)
    return round(score, 1)

def clean_json_string(s: str) -> str:
    """Clean and extract JSON from a string that might contain markdown or other text."""
    json_match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
    if json_match:
        s = json_match.group(1)
    else:
        json_match = re.search(r'`(.*?)`', s, re.DOTALL)
        if json_match:
            s = json_match.group(1)
    return s.strip()

def parse_json_response(response_text: str) -> Dict:
    """Safely parse JSON from API response."""
    try:
        cleaned_text = clean_json_string(response_text)
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {str(e)}\nText: {response_text}")
        raise ValueError(f"Failed to parse JSON response: {str(e)}")

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

def get_suggested_questions(doc_type: str, analysis: Dict) -> List[str]:
    """Generate suggested questions based on document type and content."""
    common_questions = [
        "What are the main obligations of each party?",
        "What are the termination conditions?",
        "Are there any important deadlines?",
    ]
    
    type_specific_questions = {
        "CONTRACT": [
            "What are the payment terms?",
            "What is the duration of the agreement?",
            "What are the renewal conditions?",
        ],
        "EMPLOYMENT": [
            "What are the compensation details?",
            "What are the benefits provided?",
            "What are the grounds for termination?",
        ],
        "REAL_ESTATE": [
            "What are the property details?",
            "What are the payment conditions?",
            "Are there any restrictions on use?",
        ],
    }
    
    return common_questions + type_specific_questions.get(doc_type, [])

def analyze_document(text: str, doc_type: str) -> Dict:
    try:
        # Calculate complexity score
        complexity_score = calculate_complexity_score(text)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a legal document analyzer. Respond with ONLY a JSON object in this exact format:
                {
                    "executive_summary": {
                        "main_points": ["key point 1", "key point 2"],
                        "action_items": ["action 1", "action 2"],
                        "overall_risk_level": "HIGH/MEDIUM/LOW"
                    },
                    "summary": "comprehensive summary",
                    "key_terms": [
                        {"term": "term name", "value": "term description", "category": "FINANCIAL/LEGAL/OPERATIONAL/TECHNICAL"}
                    ],
                    "dates": [
                        {"event": "event description", "date": "date value", "importance": "HIGH/MEDIUM/LOW"}
                    ],
                    "risks": [
                        {"description": "risk description", "severity": "HIGH/MEDIUM/LOW", "category": "LEGAL/FINANCIAL/OPERATIONAL"}
                    ],
                    "compliance": {
                        "missing_elements": ["missing item 1", "missing item 2"],
                        "unusual_terms": ["unusual term 1", "unusual term 2"],
                        "standard_clauses_present": ["clause 1", "clause 2"]
                    },
                    "metrics": {
                        "risk_score": 0-100,
                        "completeness_score": 0-100,
                        "clarity_score": 0-100
                    }
                }"""},
                {"role": "user", "content": f"Analyze this document:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        analysis = parse_json_response(response.choices[0].message.content)
        
        # Add complexity score and suggested questions
        analysis['document_metrics'] = {
            'complexity_score': complexity_score,
            'suggested_questions': get_suggested_questions(doc_type, analysis)
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

def detect_document_type(text: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a legal document classifier. 
                Respond with ONLY a JSON object in this exact format:
                {
                    "type": "CONTRACT/EMPLOYMENT/REAL_ESTATE/LEGAL_BRIEF/CORPORATE/PATENT/REGULATORY/FINANCE",
                    "confidence": 0.XX,
                    "indicators": ["reason 1", "reason 2"],
                    "industry": "specific industry if applicable",
                    "jurisdiction": "jurisdiction if mentioned"
                }"""},
                {"role": "user", "content": f"Classify this document:\n\n{text[:2000]}"}
            ],
            temperature=0.2
        )
        
        return parse_json_response(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Document type detection error: {str(e)}")
        return {
            "type": "CONTRACT",
            "confidence": 0.5,
            "indicators": ["Default classification due to error"]
        }

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Detect document type
        doc_type_info = detect_document_type(text)
        logger.info(f"Detected document type: {doc_type_info}")
        
        # Perform analysis
        analysis = analyze_document(text, doc_type_info["type"])
        logger.info("Analysis completed successfully")
        
        # Combine results
        result = {
            "document_type": doc_type_info,
            "analysis": analysis
        }
        
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
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
                {"role": "system", "content": """You are a legal document assistant. 
                Answer questions about the document accurately and concisely. 
                If referring to specific sections, cite them. 
                Also provide the location/page number if available."""},
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

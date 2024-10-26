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

# Document type definitions
DOCUMENT_TYPES = {
    "CONTRACT": "General contract or agreement",
    "EMPLOYMENT": "Employment-related document",
    "REAL_ESTATE": "Real estate document",
    "LEGAL_BRIEF": "Legal brief or court document",
    "CORPORATE": "Corporate document (bylaws, operating agreements)",
    "PATENT": "Patent or IP document",
    "REGULATORY": "Regulatory or compliance document",
    "FINANCE": "Financial or tax document"
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

def detect_document_type(text: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a legal document classifier. 
                Analyze the document and determine its type from these categories:
                - CONTRACT: General contract or agreement
                - EMPLOYMENT: Employment-related document
                - REAL_ESTATE: Real estate document
                - LEGAL_BRIEF: Legal brief or court document
                - CORPORATE: Corporate document
                - PATENT: Patent or IP document
                - REGULATORY: Regulatory document
                - FINANCE: Financial document
                
                Respond with a JSON object containing:
                {
                    "type": "detected type code",
                    "confidence": confidence score (0-1),
                    "indicators": ["reason 1", "reason 2"]
                }
                """},
                {"role": "user", "content": f"Classify this document:\n\n{text[:2000]}"}
            ],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Document type detection error: {str(e)}")
        return {"type": "CONTRACT", "confidence": 0.5, "indicators": ["Default classification due to error"]}

def analyze_document(text: str, doc_type: str) -> Dict:
    # Custom analysis prompts for different document types
    type_prompts = {
        "CONTRACT": """Analyze this contract for:
            1. Contract purpose and parties
            2. Key terms, obligations, and values
            3. Important dates and deadlines
            4. Termination conditions
            5. Potential risks and missing elements""",
            
        "EMPLOYMENT": """Analyze this employment document for:
            1. Position and parties involved
            2. Compensation and benefits
            3. Terms of employment
            4. Obligations and restrictions
            5. Termination conditions""",
            
        "REAL_ESTATE": """Analyze this real estate document for:
            1. Property details and parties
            2. Transaction terms
            3. Conditions and contingencies
            4. Important dates and deadlines
            5. Legal obligations and restrictions""",
            
        # Add more document types as needed
    }

    prompt = type_prompts.get(doc_type, type_prompts["CONTRACT"])
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a legal document analyzer specialized in {doc_type} documents.
                {prompt}
                
                Respond in this JSON format:
                {{
                    "summary": "comprehensive summary",
                    "key_terms": [
                        {{"term": "term name", "value": "description"}}
                    ],
                    "dates": [
                        {{"event": "event name", "date": "date value"}}
                    ],
                    "risks": ["risk 1", "risk 2"],
                    "flags": [
                        {{"severity": "HIGH/MEDIUM/LOW", "issue": "description", "recommendation": "suggestion"}}
                    ]
                }}
                """},
                {"role": "user", "content": f"Analyze this document:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        # Detect document type
        doc_type_info = detect_document_type(text)
        logger.info(f"Detected document type: {doc_type_info}")
        
        # Perform analysis
        analysis = analyze_document(text, doc_type_info["type"])
        
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
            content={"detail": f"Analysis error: {str(e)}"}
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
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the document accurately and concisely. If referring to specific sections, cite them."},
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

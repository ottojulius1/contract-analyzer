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
                }"""},
                {"role": "user", "content": f"Classify this document:\n\n{text[:2000]}"}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content
        # Try to parse the response as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {result}")
            return {
                "type": "CONTRACT",
                "confidence": 0.5,
                "indicators": ["Default classification due to parsing error"]
            }
            
    except Exception as e:
        logger.error(f"Document type detection error: {str(e)}")
        return {
            "type": "CONTRACT",
            "confidence": 0.5,
            "indicators": ["Default classification due to error"]
        }

def analyze_document(text: str, doc_type: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""You are a legal document analyzer. Analyze this document and provide:
                1. A clear summary
                2. Key terms and their values
                3. Important dates and deadlines
                4. Potential risks
                5. Areas needing attention

                Respond with a JSON object in this exact format:
                {{
                    "summary": "brief overview",
                    "key_terms": [
                        {{"term": "term name", "value": "term description"}}
                    ],
                    "dates": [
                        {{"event": "event description", "date": "date value"}}
                    ],
                    "risks": [
                        "risk description"
                    ],
                    "flags": [
                        {{"severity": "HIGH/MEDIUM/LOW", 
                          "issue": "issue description",
                          "recommendation": "suggested action"}}
                    ]
                }}"""},
                {"role": "user", "content": f"Analyze this document:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content
        # Try to parse the response as JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse analysis response as JSON: {result}")
            raise HTTPException(status_code=500, detail="Failed to parse analysis results")
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        # Log the extracted text length
        logger.info(f"Extracted {len(text)} characters from PDF")
        
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

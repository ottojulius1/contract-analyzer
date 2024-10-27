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
import difflib
from datetime import datetime
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

def compare_documents(text1: str, text2: str) -> Dict:
    """Compare two document texts and return the differences."""
    try:
        # Use difflib for initial comparison
        differ = difflib.Differ()
        diff = list(differ.compare(text1.splitlines(), text2.splitlines()))
        
        # Use AI for detailed analysis
        comparison_prompt = f"""Compare these two versions of the document and provide analysis in this JSON format:
        {{
            "summary_of_changes": "brief overview of main differences",
            "significant_changes": [
                {{"section": "section name", "change": "description of change", "impact": "HIGH/MEDIUM/LOW"}}
            ],
            "added_clauses": ["new clause 1", "new clause 2"],
            "removed_clauses": ["removed clause 1", "removed clause 2"],
            "modified_clauses": [
                {{"clause": "clause name", "original": "old text", "new": "new text", "impact": "HIGH/MEDIUM/LOW"}}
            ],
            "risk_analysis": [
                {{"risk": "description", "severity": "HIGH/MEDIUM/LOW", "recommendation": "suggested action"}}
            ]
        }}
        
        Document 1:
        {text1[:4000]}
        
        Document 2:
        {text2[:4000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document comparison expert. Analyze the differences between two versions of a document."},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0.2
        )
        
        # Parse AI analysis
        ai_analysis = json.loads(response.choices[0].message.content)
        
        # Generate line-by-line diff
        diff_analysis = {
            "additions": [line[2:] for line in diff if line.startswith('+ ')],
            "deletions": [line[2:] for line in diff if line.startswith('- ')],
            "modifications": [line[2:] for line in diff if line.startswith('? ')]
        }
        
        return {
            "ai_analysis": ai_analysis,
            "diff_analysis": diff_analysis,
            "comparison_date": datetime.now().isoformat(),
            "diff_stats": {
                "additions": len(diff_analysis["additions"]),
                "deletions": len(diff_analysis["deletions"]),
                "modifications": len(diff_analysis["modifications"])
            }
        }
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

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
                    "indicators": ["reason 1", "reason 2"],
                    "industry": "specific industry if applicable",
                    "jurisdiction": "jurisdiction if mentioned"
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
                "indicators": ["Error parsing classification"],
                "industry": "Unknown",
                "jurisdiction": "Unknown"
            }
            
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Analyze this legal document and return ONLY a JSON object without any additional text or formatting.
                Format:
                {
                    "summary": "brief overview",
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
                        {"severity": "HIGH", "issue": "description", "recommendation": "action"}
                    ],
                    "clause_analysis": {
                        "key_clauses": [
                            {"title": "clause name", "content": "clause content", "importance": "HIGH/MEDIUM/LOW"}
                        ],
                        "missing_clauses": [
                            {"clause": "missing clause name", "recommendation": "why it should be included"}
                        ],
                        "unusual_provisions": [
                            {"provision": "provision description", "explanation": "why it's unusual", "risk_level": "HIGH/MEDIUM/LOW"}
                        ]
                    }
                }"""},
                {"role": "user", "content": f"Analyze this document and identify key clauses, missing standard clauses, and any unusual provisions:\n\n{text}"}
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
                "flags": [{"severity": "HIGH", "issue": "Analysis Error", "recommendation": "Please try again"}],
                "clause_analysis": {
                    "key_clauses": [],
                    "missing_clauses": [],
                    "unusual_provisions": []
                }
            }
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "summary": "Error during analysis",
            "key_terms": [],
            "dates": [],
            "risks": ["Analysis error occurred"],
            "flags": [{"severity": "HIGH", "issue": "Analysis Error", "recommendation": "Please try again"}],
            "clause_analysis": {
                "key_clauses": [],
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

@app.post("/compare")
async def compare_documents_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    """Compare two PDF documents and return analysis of differences."""
    try:
        # Extract text from both PDFs
        contents1 = await file1.read()
        contents2 = await file2.read()
        
        text1 = extract_text_from_pdf(contents1)
        text2 = extract_text_from_pdf(contents2)
        
        if not text1.strip() or not text2.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "Could not extract text from one or both PDFs"}
            )
        
        # Get comparison results
        comparison_results = compare_documents(text1, text2)
        
        # Get individual analyses for context
        analysis1 = analyze_document(text1, "UNKNOWN")
        analysis2 = analyze_document(text2, "UNKNOWN")
        
        # Combine all results
        return JSONResponse(content={
            "comparison": comparison_results,
            "document1_analysis": analysis1,
            "document2_analysis": analysis2
        })
        
    except Exception as e:
        logger.error(f"Comparison endpoint error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Comparison failed: {str(e)}"}
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

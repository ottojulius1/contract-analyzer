from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
from typing import Dict
import PyPDF2
import io
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_analysis_prompt(text: str) -> str:
    return """Please analyze this document and provide a structured response in the following JSON format:
    {
        "document_type": {
            "type": "Contract/Agreement/Policy/etc",
            "confidence": 0.95,
            "industry": "relevant industry",
            "jurisdiction": "applicable jurisdiction",
            "indicators": ["key indicator 1", "key indicator 2"]
        },
        "analysis": {
            "summary": "Brief summary of the document",
            "key_terms": [
                {
                    "term": "term name",
                    "value": "term details",
                    "category": "FINANCIAL/LEGAL/OPERATIONAL"
                }
            ],
            "dates": [
                {
                    "date": "YYYY-MM-DD",
                    "event": "description",
                    "importance": "HIGH/MEDIUM/LOW"
                }
            ],
            "clause_analysis": {
                "clauses": [
                    {
                        "clause_name": "name",
                        "clause_text": "text"
                    }
                ],
                "missing_clauses": ["missing clause 1"],
                "unusual_provisions": ["unusual provision 1"]
            },
            "risks": [
                {
                    "risk": "description",
                    "severity": "HIGH/MEDIUM/LOW",
                    "mitigation": "suggested mitigation"
                }
            ]
        }
    }

Document text:
{text}
"""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        # Read and extract text from PDF
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page_num].extract_text()
        
        logger.debug(f"Extracted text length: {len(extracted_text)}")
        
        # Create analysis prompt
        prompt = create_analysis_prompt(extracted_text)
        
        # Get OpenAI response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document analyzer. Provide analysis in the exact JSON format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        response_text = response.choices[0].message['content']
        logger.debug(f"OpenAI response received, length: {len(response_text)}")
        
        # Parse the response and ensure it's valid JSON
        try:
            parsed_response = json.loads(response_text)
            return JSONResponse(content=parsed_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse analysis results")
            
    except Exception as e:
        logger.error(f"Error during document analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page_num].extract_text()
        
        prompt = f"""Context: {extracted_text}\n\nQuestion: {question}\n\nPlease provide a clear and concise answer based on the document content."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about documents accurately and concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        response_text = response.choices[0].message['content']
        return {"answer": response_text}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

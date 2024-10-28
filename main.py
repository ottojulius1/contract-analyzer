from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_analysis_prompt(text: str) -> str:
    return """You are an expert legal document analyzer with deep experience in contract analysis, legal review, and risk assessment. 
    Analyze the provided document thoroughly and provide a comprehensive structured response in the following JSON format.
    
    Requirements for each section:

    1. Document Type:
       - Provide specific document classification with high confidence
       - Identify industry sector and jurisdiction
       - List key indicators that support the classification

    2. Summary:
       - Write a detailed 3-4 paragraph summary
       - Cover main purposes, key obligations, and important terms
       - Highlight unique or notable aspects

    3. Key Terms:
       - Identify at least 8-10 significant terms
       - Include financial terms, deadlines, obligations
       - Categorize accurately (FINANCIAL/LEGAL/OPERATIONAL)
       - Provide detailed explanations for each term

    4. Important Dates:
       - List ALL dates mentioned (at least 5-6 if present)
       - Include deadlines, renewal dates, termination dates
       - Clearly explain the significance of each date
       - Rate importance (HIGH/MEDIUM/LOW)

    5. Action Items:
       - List at least 5 required actions
       - Include deadlines and responsible parties
       - Prioritize by urgency
       - Include compliance requirements

    6. Clause Analysis:
       - Analyze at least 6-8 key clauses
       - Provide detailed explanation of each clause
       - Highlight unusual or non-standard language
       - Identify missing standard clauses

    7. Risks Assessment:
       - Identify at least 5 potential risks
       - Rate severity (HIGH/MEDIUM/LOW)
       - Provide specific mitigation strategies
       - Consider business, legal, and operational risks

    Response Format:
    {
        "document_type": {
            "type": "detailed document type",
            "confidence": confidence_score,
            "industry": "specific industry",
            "jurisdiction": "applicable jurisdiction",
            "indicators": ["detailed indicator 1", "detailed indicator 2", ...]
        },
        "analysis": {
            "summary": "detailed multi-paragraph summary",
            "key_terms": [
                {
                    "term": "term name",
                    "value": "detailed explanation",
                    "category": "FINANCIAL/LEGAL/OPERATIONAL",
                    "importance": "HIGH/MEDIUM/LOW"
                }
            ],
            "dates": [
                {
                    "date": "YYYY-MM-DD",
                    "event": "detailed description",
                    "importance": "HIGH/MEDIUM/LOW",
                    "implications": "specific implications"
                }
            ],
            "action_items": [
                {
                    "action": "specific action required",
                    "deadline": "deadline if applicable",
                    "priority": "HIGH/MEDIUM/LOW",
                    "responsible_party": "who needs to act",
                    "details": "additional details"
                }
            ],
            "clause_analysis": {
                "clauses": [
                    {
                        "clause_name": "name of clause",
                        "clause_text": "relevant text",
                        "analysis": "detailed analysis",
                        "implications": "business/legal implications",
                        "risk_level": "HIGH/MEDIUM/LOW"
                    }
                ],
                "missing_clauses": ["detailed missing clause 1", "detailed missing clause 2"],
                "unusual_provisions": ["detailed unusual provision 1", "detailed unusual provision 2"]
            },
            "risks": [
                {
                    "risk": "detailed risk description",
                    "severity": "HIGH/MEDIUM/LOW",
                    "impact": "specific impact description",
                    "likelihood": "HIGH/MEDIUM/LOW",
                    "mitigation": "detailed mitigation strategy"
                }
            ]
        }
    }

    Document text to analyze:
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Using 16k model for longer context
            messages=[
                {"role": "system", "content": "You are an expert legal document analyzer. Provide comprehensive analysis in the exact JSON format requested, ensuring all sections are detailed and complete."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000  # Increased token limit for more detailed response
        )
        
        response_text = response.choices[0].message.content
        logger.debug(f"OpenAI response received, length: {len(response_text)}")
        
        try:
            parsed_response = json.loads(response_text)
            return JSONResponse(content=parsed_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
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
        
        prompt = f"""Context: {extracted_text}

Question: {question}

Instructions:
1. Carefully analyze the document context
2. Provide a detailed, accurate answer based solely on the document content
3. If the information isn't in the document, say so
4. Include specific references or quotes where relevant
5. Break down complex answers into clear points

Please provide a clear, thorough answer to the question."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert document analyst providing accurate, detailed answers based solely on document content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        response_text = response.choices[0].message.content
        return {"answer": response_text}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

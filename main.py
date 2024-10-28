from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_analysis_prompt(text: str) -> str:
    return """You are an expert legal document analyzer. First identify the exact type of legal document, then provide a detailed analysis based on its actual content.

STEP 1: Document Classification
First, identify the specific type of document (e.g., Retainer Agreement, Employment Contract, Settlement Agreement, Contract, Lease, etc.) and its key identifiers.

STEP 2: Detailed Analysis
Based on the identified document type, extract and analyze the following FROM THE ACTUAL DOCUMENT TEXT:

{
    "document_type": {
        "type": "QUOTE the exact document title/type as written",
        "category": "Legal category (e.g., Corporate, Family Law, Real Estate, etc.)",
        "jurisdiction": "QUOTE any jurisdiction mentioned",
        "parties": {
            "party1": {
                "name": "EXACT name as written",
                "role": "Role as specified in document"
            },
            "party2": {
                "name": "EXACT name as written",
                "role": "Role as specified in document"
            }
        },
        "matter": "QUOTE the specific matter or purpose",
        "date": "QUOTE document date"
    },
    "analysis": {
        "summary": "Write a SPECIFIC summary that includes:
            1. Exact purpose of this document
            2. Actual names of all parties
            3. All monetary values mentioned
            4. Key dates and deadlines
            5. Main obligations
            NO GENERIC DESCRIPTIONS - use actual details from document",
        
        "key_terms": [
            {
                "term": "QUOTE the exact term/section heading",
                "content": "QUOTE relevant text",
                "value": "For monetary terms, QUOTE exact amounts",
                "category": "FINANCIAL/LEGAL/OPERATIONAL",
                "location": "Section or page reference"
            }
        ],
        
        "monetary_provisions": [
            {
                "type": "Type of payment/fee/amount",
                "amount": "QUOTE exact amount",
                "details": "QUOTE relevant text",
                "location": "Section reference"
            }
        ],
        
        "dates_and_deadlines": [
            {
                "date": "QUOTE exact date",
                "event": "QUOTE what happens on this date",
                "significance": "HIGH/MEDIUM/LOW",
                "details": "QUOTE relevant text"
            }
        ],
        
        "key_provisions": [
            {
                "title": "QUOTE section/provision title",
                "text": "QUOTE actual provision text",
                "significance": "Explain significance using details from document",
                "location": "Section reference"
            }
        ],
        
        "obligations": {
            "party1": [
                "LIST specific obligations QUOTED from document"
            ],
            "party2": [
                "LIST specific obligations QUOTED from document"
            ]
        },
        
        "critical_terms": [
            {
                "term": "QUOTE important term",
                "explanation": "Explain using ACTUAL document content",
                "importance": "HIGH/MEDIUM/LOW"
            }
        ],
        
        "unusual_provisions": [
            {
                "provision": "QUOTE any unusual/notable provisions",
                "analysis": "Explain why notable",
                "location": "Section reference"
            }
        ],
        
        "risks": [
            {
                "risk": "Identify specific risk based on document content",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "QUOTE relevant text",
                "mitigation": "If mentioned in document, QUOTE mitigation measures"
            }
        ]
    }
}

IMPORTANT RULES:
1. NEVER make up or infer information
2. ONLY use text actually present in the document
3. Use "Not specified" if information isn't in the document
4. QUOTE actual document text wherever possible
5. Include section/page references
6. Focus on what makes this specific document unique
7. Adapt analysis based on document type
8. If monetary values, dates, or specific terms exist, always include them
9. Identify unusual or notable provisions
10. Note any missing standard provisions for this type of document

Remember: This tool will be used for ALL types of legal documents, so first identify the type, then provide appropriate analysis for that specific type of document.

Document to analyze:
{text}"""
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            extracted_text += f"\n{page_text}"
        
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst. Provide detailed, specific analysis based solely on the document content."
                },
                {"role": "user", "content": create_analysis_prompt(extracted_text)}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        logger.debug(f"Received analysis response of {len(response_text)} characters")
        
        try:
            parsed_response = json.loads(response_text)
            return JSONResponse(content=parsed_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
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

Provide a focused answer that:
1. Cites specific sections of the document
2. Uses actual details and quotes when relevant
3. Explains any legal implications
4. Stays strictly within the document's content"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst providing detailed answers about legal documents."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        return {"answer": response_text}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

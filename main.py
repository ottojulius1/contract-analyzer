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
    return """You are an expert legal document analyzer. Analyze this document and provide specific details with exact quotes and references.

Required Format:
{
    "document_type": {
        "type": "QUOTE exact document title/type",
        "category": "QUOTE legal category (e.g., Family Law, Corporate, etc.)",
        "jurisdiction": "QUOTE jurisdiction",
        "parties": {
            "party1": {
                "name": "QUOTE exact name",
                "role": "QUOTE role"
            },
            "party2": {
                "name": "QUOTE exact name",
                "role": "QUOTE role"
            }
        },
        "matter": "QUOTE specific matter/case reference"
    },
    "analysis": {
        "summary": "SPECIFIC summary using only actual document content, including: 1) Document purpose, 2) Party names and roles, 3) Key monetary values, 4) Important dates, 5) Main obligations",
        "key_terms": [
            {
                "term": "QUOTE section name/term",
                "content": "QUOTE relevant text",
                "value": "QUOTE monetary value if applicable",
                "category": "FINANCIAL/LEGAL/OPERATIONAL"
            }
        ],
        "dates_and_deadlines": [
            {
                "date": "QUOTE exact date",
                "event": "QUOTE related event/requirement",
                "significance": "HIGH/MEDIUM/LOW",
                "details": "QUOTE relevant text"
            }
        ],
        "key_provisions": [
            {
                "title": "QUOTE provision name",
                "text": "QUOTE actual provision text",
                "significance": "Explain using document content"
            }
        ],
        "risks": [
            {
                "risk": "Describe risk using document content",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "QUOTE relevant text"
            }
        ]
    }
}

REQUIREMENTS:
1. Use EXACT QUOTES from the document
2. Include ALL monetary values found
3. Include ALL dates and deadlines
4. Reference specific sections
5. NO generic text - use actual document content
6. If information isn't in document, mark as "Not specified"

Document text to analyze:
{text}"""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            extracted_text += f"\nPage {page_num + 1}:\n{page_text}"
        
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst. Extract and quote specific details from the document."
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
            extracted_text += f"\nPage {page_num + 1}:\n{pdf_reader.pages[page_num].extract_text()}"
        
        prompt = f"""Document text: {extracted_text}

Question: {question}

Provide a detailed answer that:
1. QUOTES specific sections of the document
2. References exact page/section numbers
3. Uses only information from the document
4. Explains any legal implications found in the document"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst. Quote and reference specific parts of the document in your answers."
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

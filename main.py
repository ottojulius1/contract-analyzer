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
    base_prompt = """Analyze this legal document and provide a JSON response that MUST include actual content from the document, not template text.

Required JSON Structure:
{
    "document_type": {
        "type": "Exact document title/type from document",
        "category": "Legal category from document (e.g., Family Law, Corporate)",
        "jurisdiction": "Jurisdiction explicitly mentioned in document",
        "parties": {
            "party1": {
                "name": "First party's exact name from document",
                "role": "First party's role from document"
            },
            "party2": {
                "name": "Second party's exact name from document",
                "role": "Second party's role from document"
            }
        },
        "matter": "Specific case/matter reference from document"
    },
    "analysis": {
        "summary": "Detailed summary using actual document content with: 1) Document exact purpose, 2) Party names and roles, 3) All monetary values mentioned, 4) Important dates found, 5) Main obligations stated",
        "key_terms": [
            {
                "term": "Term heading/name from document",
                "content": "Exact quote of relevant text",
                "value": "Monetary amount if mentioned",
                "category": "FINANCIAL/LEGAL/OPERATIONAL"
            }
        ],
        "dates_and_deadlines": [
            {
                "date": "Actual date from document",
                "event": "Event description from document",
                "significance": "HIGH/MEDIUM/LOW",
                "details": "Quote of relevant text"
            }
        ],
        "key_provisions": [
            {
                "title": "Actual provision name from document",
                "text": "Direct quote of provision text",
                "significance": "Explanation using document content"
            }
        ],
        "risks": [
            {
                "risk": "Risk identified from document content",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "Quote supporting this risk"
            }
        ]
    }
}

CRITICAL REQUIREMENTS:
1. Use ONLY content found in the document
2. Include ALL monetary values mentioned
3. Include ALL dates and deadlines found
4. Quote specific sections when possible
5. NEVER use placeholder or template text
6. Use "Not specified" for truly missing information

Document to analyze:
"""
    return base_prompt + text

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

        system_message = """You are an expert legal document analyzer. Your task is to:
1. Extract specific information from legal documents
2. Quote relevant text directly
3. Identify key terms, dates, and provisions
4. Return analysis in valid JSON format
5. Never include placeholders or template text in response
6. Always use actual document content
7. Maintain proper JSON structure
8. Include all monetary values and dates found"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": create_analysis_prompt(extracted_text)}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        logger.debug(f"Received analysis response")
        
        try:
            # Clean and parse JSON response
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            parsed_response = json.loads(cleaned_response)
            logger.debug("Successfully parsed JSON response")
            return JSONResponse(content=parsed_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse analysis results. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error during document analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing document: {str(e)}"
        )

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

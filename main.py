from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import PyPDF2
import io
import os
import json
import logging
import datetime

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
    return """You are an expert legal document analyzer. Analyze this document comprehensively and provide detailed insights.

Required Analysis Structure:
{
    "document_type": {
        "type": "QUOTE exact document title/type",
        "category": "Legal category (e.g., Family Law, Corporate)",
        "jurisdiction": "QUOTE jurisdiction mentioned",
        "parties": [
            {
                "name": "QUOTE exact name",
                "role": "QUOTE role in document"
            }
        ],
        "matter": "QUOTE specific matter/case reference"
    },
    "analysis": {
        "summary": "Detailed summary including: 1) Document purpose, 2) Key parties and roles, 3) Main obligations, 4) Important dates, 5) Critical terms - USE ACTUAL DOCUMENT CONTENT",
        
        "key_terms": [
            {
                "term": "QUOTE term from document",
                "content": "QUOTE relevant text",
                "value": "QUOTE monetary value if applicable",
                "category": "FINANCIAL/LEGAL/OPERATIONAL",
                "location": "Section reference"
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
                "significance": "Explain significance using document content",
                "location": "Section reference"
            }
        ],
        
        "obligations": {
            "party1": [
                "QUOTE specific obligations from document"
            ],
            "party2": [
                "QUOTE specific obligations from document"
            ]
        },
        
        "risks": [
            {
                "risk": "Identify specific risk from document content",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "QUOTE relevant text",
                "mitigation": "QUOTE any mitigation measures mentioned"
            }
        ],
        
        "next_steps": [
            {
                "action": "Specific required action",
                "timeline": "When it must be done",
                "responsible_party": "Who must do it",
                "source": "Section reference"
            }
        ]
    }
}

CRITICAL REQUIREMENTS:
1. Use ACTUAL QUOTES from the document
2. Provide section references where possible
3. Include ALL monetary values found
4. Include ALL dates and deadlines
5. Identify ALL key terms and provisions
6. Note both explicit and implicit obligations
7. Focus on SPECIFIC details, not generic descriptions
8. Quote actual text to support analysis

Document to analyze:
"""
    return base_prompt + text

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            extracted_text += f"\nPage {page_num + 1}:\n{page_text}"
        
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF")

        # System message for legal analysis
        system_message = """You are an expert legal document analyzer. Your task is to:
1. Extract and quote specific provisions
2. Identify all key terms, dates, and requirements
3. Note both explicit and implicit obligations
4. Flag any unusual or missing elements
5. Provide section-specific references
6. Focus on practical implications
7. Highlight all critical deadlines"""

        # Make API request
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
        logger.debug("Received response from OpenAI")
        
        try:
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            parsed_response = json.loads(cleaned_response)
            
            # Add metadata
            parsed_response["analysis_metadata"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "document_length": len(extracted_text),
                "analysis_version": "2.0",
                "document_name": file.filename
            }

            return JSONResponse(content=parsed_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            raise HTTPException(status_code=500, detail="Failed to parse analysis results")
            
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

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
1. Quotes specific sections of the document
2. References exact page/section numbers
3. Explains in plain language
4. Notes any legal implications
5. Highlights practical impacts

Base your answer ONLY on the document's actual content."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst. Provide detailed, accurate answers with specific references to the document."
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

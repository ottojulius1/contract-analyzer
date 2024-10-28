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
    return """You are conducting a legal document analysis. You MUST:
1. Extract and quote ACTUAL text from the document
2. Use REAL names, dates, numbers, and terms found in the document
3. DO NOT generate generic descriptions
4. Every point must cite specific details from the document

Analyze this legal document and provide a structured response that includes ONLY ACTUAL content from the document:
{
    "document_type": {
        "type": "[QUOTE exact title/heading of document]",
        "industry": "[EXTRACT actual industry references from document]",
        "jurisdiction": "[QUOTE jurisdiction clause or location references]",
        "parties": [
            "[LIST actual party names as written in document]"
        ]
    },
    "analysis": {
        "summary": "[EXTRACT KEY DETAILS ONLY: 1) Exact names of all parties 2) Specific service/employment/subject matter described 3) Actual monetary values and payment terms 4) Real dates and deadlines 5) Quote unique or important clauses. NO GENERIC DESCRIPTIONS.]",
        "key_terms": [
            {
                "term": "[QUOTE section heading or key term]",
                "value": "[QUOTE actual clause text defining this term]",
                "category": "FINANCIAL/LEGAL/OPERATIONAL",
                "source": "[Reference section number or location]"
            }
        ],
        "dates": [
            {
                "date": "[QUOTE actual date from document]",
                "event": "[DESCRIBE specific event using document language]",
                "importance": "HIGH/MEDIUM/LOW",
                "source": "[Reference section number or location]"
            }
        ],
        "clause_analysis": {
            "clauses": [
                {
                    "clause_name": "[QUOTE actual clause heading]",
                    "clause_text": "[QUOTE exact clause text]",
                    "analysis": "[Explain implications using specifics from the clause]",
                    "location": "[Section number or location]"
                }
            ],
            "missing_clauses": [
                "[List standard clauses that are actually missing, be specific]"
            ],
            "unusual_provisions": [
                "[QUOTE any non-standard clauses found, with section references]"
            ]
        },
        "risks": [
            {
                "risk": "[Describe specific risk based on actual clause content]",
                "severity": "HIGH/MEDIUM/LOW",
                "source": "[Quote relevant clause text]",
                "mitigation": "[Specific mitigation based on document terms]"
            }
        ]
    }
}

IMPORTANT:
- Every field must contain ACTUAL content from the document
- Use quotes and section references
- Do not generate generic text
- If information isn't in the document, say "Not specified in document" rather than making assumptions
- Focus on unique aspects of this specific document

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

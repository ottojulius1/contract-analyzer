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
    return """Analyze this legal document and provide a detailed response in the following JSON format:
    {
        "document_type": {
            "type": "specific document type",
            "industry": "specific industry sector",
            "jurisdiction": "governing jurisdiction",
            "parties": ["list all parties involved"]
        },
        "analysis": {
            "summary": "Provide a detailed, specific summary of this exact document including: 1) What exactly this document does 2) The specific parties and their roles 3) Key terms, payments, or monetary values 4) Important dates or deadlines 5) Main obligations of each party. Use actual details from the document.",
            "key_terms": [
                {
                    "term": "specific term from document",
                    "value": "actual details and implications",
                    "category": "FINANCIAL/LEGAL/OPERATIONAL"
                }
            ],
            "dates": [
                {
                    "date": "actual date",
                    "event": "specific event or deadline",
                    "importance": "HIGH/MEDIUM/LOW"
                }
            ],
            "clause_analysis": {
                "clauses": [
                    {
                        "clause_name": "name of specific clause",
                        "clause_text": "actual text from document",
                        "analysis": "implications and importance"
                    }
                ],
                "missing_clauses": ["important standard clauses that are missing"],
                "unusual_provisions": ["any unusual or noteworthy provisions"]
            },
            "risks": [
                {
                    "risk": "specific risk identified",
                    "severity": "HIGH/MEDIUM/LOW",
                    "mitigation": "specific mitigation strategy"
                }
            ]
        }
    }

Document text:
{text}

Important instructions:
1. Be specific - use actual details from the document
2. Never provide generic descriptions
3. Include all monetary values, dates, and specific terms found
4. Quote actual clause text when relevant
5. For the summary, focus on what makes this document unique and important"""

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

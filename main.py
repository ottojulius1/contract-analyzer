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
    base_prompt = """Analyze this legal document and provide a JSON response with the following structure:
{
    "document_type": {
        "type": "[Document Type]",
        "category": "[Legal Category]",
        "jurisdiction": "[Jurisdiction]",
        "parties": {
            "party1": {
                "name": "[Party Name]",
                "role": "[Party Role]"
            },
            "party2": {
                "name": "[Party Name]",
                "role": "[Party Role]"
            }
        },
        "matter": "[Matter/Case Reference]"
    },
    "analysis": {
        "summary": "[Document Summary]",
        "key_terms": [
            {
                "term": "[Term Name]",
                "content": "[Term Content]",
                "value": "[Monetary Value if applicable]",
                "category": "FINANCIAL/LEGAL/OPERATIONAL"
            }
        ],
        "dates_and_deadlines": [
            {
                "date": "[Date]",
                "event": "[Event Description]",
                "significance": "HIGH/MEDIUM/LOW",
                "details": "[Event Details]"
            }
        ],
        "key_provisions": [
            {
                "title": "[Provision Title]",
                "text": "[Provision Text]",
                "significance": "[Significance]"
            }
        ],
        "risks": [
            {
                "risk": "[Risk Description]",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "[Risk Basis]"
            }
        ]
    }
}

Instructions:
1. Replace all text in [] with actual content from the document
2. Use exact quotes for important text
3. Include all monetary values found
4. Include all dates and deadlines
5. Reference specific sections
6. Use only information from the document
7. For any missing information, use "Not specified"
8. Return only valid JSON

Analyze this document:
"""
    return base_prompt + text

@app.post("/analyze")
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
        
        # Create system message
        system_message = """You are an expert legal document analyzer. Your task is to:
1. Extract specific information from legal documents
2. Quote relevant text directly
3. Identify key terms, dates, and provisions
4. Return analysis in valid JSON format
5. Never include placeholders or template text in response
6. Always use actual document content"""
        
        # Send request to OpenAI
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
            # Try to clean and parse JSON
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            parsed_response = json.loads(cleaned_response)
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

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
import time
from typing import Dict, List, Optional, Union

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

def num_tokens_from_string(string: str) -> int:
    """Estimate the number of tokens in a string."""
    return len(string.split()) + len(string) // 4

def chunk_document(text: str, max_tokens: int = 1500) -> List[str]:
    """Split document into very small chunks to avoid token limits."""
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = num_tokens_from_string(paragraph)
        
        # If adding this paragraph would exceed the limit
        if current_tokens + paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_tokens = paragraph_tokens
        else:
            if current_chunk:
                current_chunk += '\n\n'
            current_chunk += paragraph
            current_tokens += paragraph_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_chunk_with_retry(chunk: str, system_message: str, attempt: int = 0) -> Optional[Dict]:
    """Process a chunk with retry logic and rate limiting."""
    max_attempts = 3
    try:
        # Add delay between attempts
        if attempt > 0:
            time.sleep(attempt * 2)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": create_analysis_prompt(chunk)}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing chunk (attempt {attempt + 1}): {str(e)}")
        if attempt < max_attempts - 1:
            return process_chunk_with_retry(chunk, system_message, attempt + 1)
return None

def create_analysis_prompt(text: str) -> str:
    """Create a minimal but effective analysis prompt."""
    base_prompt = '''You are a legal expert. Analyze this document section and provide information in this exact format:

{
    "document_type": {
        "type": "Document type",
        "category": "Category",
        "jurisdiction": "Jurisdiction",
        "matter": "Subject matter",
        "parties": [
            { "name": "Party name", "role": "Party role" }
        ]
    },
    "analysis": {
        "summary": "Brief but detailed section summary",
        "key_terms": [
            {
                "term": "Key term found",
                "content": "Brief description",
                "category": "FINANCIAL/LEGAL/OPERATIONAL",
                "location": "Section reference"
            }
        ],
        "dates_and_deadlines": [
            {
                "date": "Specific date",
                "event": "What happens",
                "details": "Important details"
            }
        ],
        "key_provisions": [
            {
                "title": "Provision name",
                "text": "Brief quote",
                "significance": "Why important"
            }
        ],
        "risks": [
            {
                "risk": "Risk identified",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "Why this is a risk"
            }
        ],
        "next_steps": [
            {
                "action": "What needs to be done",
                "timeline": "When to do it"
            }
        ],
        "obligations": {
            "party1": [],
            "party2": []
        }
    }
}
    return base_prompt + "\n\nDocument text to analyze:\n" + text
{{
    "document_type": {{
        "type": "Document type",
        "category": "Category",
        "jurisdiction": "Jurisdiction",
        "matter": "Subject matter",
        "parties": [
            {{ "name": "Party name", "role": "Party role" }}
        ]
    }},
    "analysis": {{
        "summary": "Brief but detailed section summary",
        "key_terms": [
            {{
                "term": "Key term found",
                "content": "Brief description",
                "category": "FINANCIAL/LEGAL/OPERATIONAL",
                "location": "Section reference"
            }}
        ],
        "dates_and_deadlines": [
            {{
                "date": "Specific date",
                "event": "What happens",
                "details": "Important details"
            }}
        ],
        "key_provisions": [
            {{
                "title": "Provision name",
                "text": "Brief quote",
                "significance": "Why important"
            }}
        ],
        "risks": [
            {{
                "risk": "Risk identified",
                "severity": "HIGH/MEDIUM/LOW",
                "basis": "Why this is a risk"
            }}
        ],
        "next_steps": [
            {{
                "action": "What needs to be done",
                "timeline": "When to do it"
            }}
        ],
        "obligations": {{
            "party1": [],
"party2": []
        }}
    }}
}}'''
    return base_prompt + text

def merge_analyses(analyses: List[Dict]) -> Dict:
    """Merge analyses while avoiding duplicates."""
    merged = {
        "document_type": {
            "type": "",
            "category": "",
            "jurisdiction": "",
            "matter": "",
            "parties": []
        },
        "analysis": {
            "summary": "",
            "key_terms": [],
            "dates_and_deadlines": [],
            "key_provisions": [],
            "risks": [],
            "next_steps": [],
            "obligations": {
                "party1": [],
                "party2": []
            }
        }
    }
    # Update document type from first valid analysis
    for analysis in analyses:
        if analysis and analysis.get("document_type"):
            merged["document_type"].update(analysis["document_type"])
            if analysis["document_type"].get("parties"):
                merged["document_type"]["parties"] = analysis["document_type"]["parties"]
            break
    
    # Merge all analyses
    summaries = []
    for analysis in analyses:
        if not analysis or "analysis" not in analysis:
            continue
            
        curr = analysis["analysis"]
        summaries.append(curr.get("summary", ""))
        
        for key in ["key_terms", "dates_and_deadlines", "key_provisions", "risks", "next_steps"]:
            if key in curr and curr[key]:
                merged["analysis"][key].extend(curr[key])
        
        if "obligations" in curr:
            if curr["obligations"].get("party1"):
                merged["analysis"]["obligations"]["party1"].extend(curr["obligations"]["party1"])
            if curr["obligations"].get("party2"):
                merged["analysis"]["obligations"]["party2"].extend(curr["obligations"]["party2"])
    
    # Combine summaries
    merged["analysis"]["summary"] = " ".join(summaries)
    
    # Remove duplicates
    for key in ["key_terms", "dates_and_deadlines", "key_provisions", "risks", "next_steps"]:
        merged["analysis"][key] = list({
            json.dumps(item): item 
            for item in merged["analysis"][key]
        }.values())
    
    merged["analysis"]["obligations"]["party1"] = list(set(merged["analysis"]["obligations"]["party1"]))
    merged["analysis"]["obligations"]["party2"] = list(set(merged["analysis"]["obligations"]["party2"]))
    
    return merged

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
        
        # Split into small chunks
        chunks = chunk_document(extracted_text)
        logger.debug(f"Split document into {len(chunks)} chunks")
        
        system_message = """You are a legal document analyzer. Focus on:
1. Identifying key terms and provisions
2. Finding dates and deadlines
3. Spotting risks and obligations
4. Noting action items
Provide specific quotes and references."""

        # Process chunks with delay between each
        analyses = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            if i > 0:
                time.sleep(1)  # Delay between chunks
            result = process_chunk_with_retry(chunk, system_message)
            if result:
                analyses.append(result)

        # Merge and return results
        merged_analysis = merge_analyses(analyses)
        
        merged_analysis["analysis_metadata"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "document_length": len(extracted_text),
            "analysis_version": "4.0",
            "document_name": file.filename,
            "chunks_processed": len(chunks),
            "successful_chunks": len(analyses)
        }

        return JSONResponse(content=merged_analysis)
            
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += f"\nPage {page_num + 1}:\n{pdf_reader.pages[page_num].extract_text()}"

        # Split into small chunks
        chunks = chunk_document(extracted_text, max_tokens=1500)
        logger.debug(f"Split document into {len(chunks)} chunks for Q&A")
        
        answers = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(1)  # Rate limiting
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a legal expert. Answer questions based only on the document content."
                        },
                        {
                            "role": "user",
                            "content": f"Document section:\n{chunk}\n\nQuestion: {question}\n\nProvide relevant information with specific quotes."
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                if answer and not answer.lower().startswith(("i don't", "no relevant", "i cannot")):
                    answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing Q&A chunk {i+1}: {str(e)}")
                continue
        
        if not answers:
            return {"answer": "I couldn't find relevant information to answer your question in the document."}
        
        combined_answer = " ".join(answers)
        
        return {"answer": combined_answer}
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

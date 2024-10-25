from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import openai
import PyPDF2
import io
import os
import json

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contract-analyzer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class AnalysisResponse(BaseModel):
    summary: str
    key_terms: List[dict]
    risks: List[str]

@app.get("/")
async def root():
    return {"message": "Contract Analyzer API is running"}

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only PDF files are supported"}
        )
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Convert to text
        try:
            pdf = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Error reading PDF: {str(e)}"}
            )

        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "No text could be extracted from the PDF"}
            )

        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a legal document analyzer. Analyze the contract and provide:
                        1. A clear summary of the contract's purpose and main points
                        2. Key terms and their values or implications
                        3. Potential risks, missing elements, or areas needing attention
                        
                        Format your response as a JSON object with these exact keys:
                        {
                            "summary": "brief summary here",
                            "key_terms": [{"term": "term name", "value": "term value"}],
                            "risks": ["risk 1", "risk 2"]
                        }
                        """
                    },
                    {"role": "user", "content": f"Analyze this contract:\n\n{text}"}
                ],
                temperature=0.2
            )
            
            # Parse OpenAI response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            return JSONResponse(content=result)

        except openai.error.AuthenticationError:
            return JSONResponse(
                status_code=500,
                content={"detail": "OpenAI API key is invalid"}
            )
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=500,
                content={"detail": "Error parsing AI response"}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"AI analysis error: {str(e)}"}
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing request: {str(e)}"}
        )

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question is required"}
        )
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Convert to text
        pdf = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

        # Call OpenAI API for Q&A
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the contract accurately and concisely."},
                {"role": "user", "content": f"Contract text: {text}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        
        return JSONResponse(content={"answer": response.choices[0].message.content})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing question: {str(e)}"}
        )

@app.options("/analyze")
async def analyze_options():
    return JSONResponse(content={})

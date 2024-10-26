from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import PyPDF2
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://contract-analyzer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def root():
    return {"message": "Contract Analyzer API is running"}

def extract_text_from_pdf(file_bytes):
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    try:
        # Read and extract text from PDF
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"detail": "No text could be extracted from the PDF"}
            )
        
        # Analyze with OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a legal document analyzer. Analyze the provided contract and extract:
                    1. A clear summary of the contract's purpose and main points
                    2. Key terms and their specific values or implications
                    3. Potential risks, missing elements, or areas needing attention
                    
                    Respond in this exact JSON format:
                    {
                        "summary": "comprehensive summary here",
                        "key_terms": [
                            {"term": "term name", "value": "specific value or description"}
                        ],
                        "risks": ["risk 1", "risk 2"]
                    }
                    """},
                {"role": "user", "content": f"Analyze this contract:\n\n{text}"}
            ],
            temperature=0.2
        )
        
        # Get the response content
        analysis = response.choices[0].message.content
        
        # Return the analysis
        return JSONResponse(content=eval(analysis))

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Analysis error: {str(e)}"}
        )

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    if not question:
        return JSONResponse(
            status_code=400,
            content={"detail": "Question is required"}
        )
    
    try:
        # Read and extract text
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        # Ask OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document assistant. Answer questions about the contract accurately and concisely."},
                {"role": "user", "content": f"Contract text: {text}\n\nQuestion: {question}"}
            ],
            temperature=0.3
        )
        
        return JSONResponse(content={"answer": response.choices[0].message.content})
        
    except Exception as e:
        logger.error(f"Question error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Question error: {str(e)}"}
        )

@app.options("/analyze")
async def analyze_options():
    return JSONResponse(content={})

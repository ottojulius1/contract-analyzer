from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
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

@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    
    try:
        # Test OpenAI connection
        logger.info("Testing OpenAI connection...")
        try:
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Test message"}
                ]
            )
            logger.info("OpenAI connection successful")
        except Exception as e:
            logger.error(f"OpenAI connection failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"OpenAI API error: {str(e)}"}
            )

        # If we get here, OpenAI is working
        return JSONResponse(content={
            "summary": "Test summary - OpenAI connection working",
            "key_terms": [
                {"term": "Test Term", "value": "Test Value"}
            ],
            "risks": ["Test Risk"]
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error: {str(e)}"}
        )

@app.options("/analyze")
async def analyze_options():
    return JSONResponse(content={})

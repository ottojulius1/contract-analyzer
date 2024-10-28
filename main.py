from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
from typing import Dict
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

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page_num].extract_text()

        logger.debug(f"Extracted text: {extracted_text[:1000]}...")  # Logs first 1000 characters for debugging

        prompt = f"Analyze this legal document:\n\n{extracted_text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analyze the document and return a detailed legal analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.2
        )

        response_text = response.choices[0].message['content']
        logger.debug(f"OpenAI response: {response_text}")

        return JSONResponse(content=json.loads(response_text))

    except Exception as e:
        logger.error(f"Error during document analysis: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing document")


@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            extracted_text += pdf_reader.pages[page_num].extract_text()

        logger.debug(f"Extracted text: {extracted_text[:1000]}...")

        prompt = f"Document Text:\n{extracted_text}\n\nQuestion: {question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer questions based on the document."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )

        response_text = response.choices[0].message['content']
        logger.debug(f"OpenAI response to question: {response_text}")

        return {"answer": response_text}

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")

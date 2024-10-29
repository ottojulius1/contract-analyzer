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
from typing import Dict, List, Optional

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
    base_prompt = """You are a highly experienced legal professional conducting a thorough document analysis. Think like both an expert lawyer and a client advocate. Your analysis must be comprehensive, detailed, and insightful.

ANALYSIS APPROACH:
1. First, thoroughly read and understand the entire document
2. Identify ALL significant elements, both explicit and implicit
3. Consider both legal and practical implications
4. Flag any unusual, missing, or concerning elements
5. Note relationships between different provisions
6. Highlight both rights and obligations of all parties
Required Analysis Structure:
{
    "document_profile": {
        "classification": {
            "document_type": "Exact document type",
            "legal_category": "Area of law",
            "jurisdiction": "Governing jurisdiction",
            "governing_law": "Specific laws/regulations that apply",
            "document_subtype": "Specific type within category",
            "applicable_regulations": ["List of relevant regulations"]
        },
        "parties": [
            {
                "name": "Exact party name",
                "role": "Role in document",
                "type": "individual/entity",
                "primary_obligations": ["List key obligations"],
                "primary_rights": ["List key rights"],
                "key_restrictions": ["List main restrictions/limitations"],
                "address": "Full address if provided",
                "relationship_to_others": "Relationship to other parties",
                "authority": "Authority/capacity in which they act"
            }
        ],
        "matter_info": {
            "subject": "Specific subject matter",
            "purpose": "Document's primary purpose",
            "scope": {
                "included": ["What's explicitly included"],
                "excluded": ["What's explicitly excluded"],
                "conditional": ["What's conditionally included"],
                "geographic_scope": "Territorial scope if specified",
                "temporal_scope": "Time period covered"
            },
            "special_circumstances": ["Any unique aspects"],
            "related_matters": ["Related cases/documents"],
            "precedent_documents": ["Reference documents"]
        }
    },
    "comprehensive_summary": {
        "executive_brief": "Clear, detailed explanation of document purpose and effect",
        "key_points": [
            {
                "point": "Major point from document",
                "explanation": "Plain language explanation",
                "legal_significance": "Legal implications",
                "practical_impact": "Real-world impact",
                "source": "Section reference",
                "related_provisions": ["Related sections"],
                "urgency_level": "HIGH/MEDIUM/LOW"
            }
        ],
        "unusual_aspects": [
            {
                "aspect": "What's unusual",
                "why_significant": "Why it matters",
                "potential_impact": "Possible consequences",
                "comparison": "How it differs from standard",
                "recommendations": ["Suggested approaches"]
            }
        ],
        "critical_elements": [
            {
                "element": "Critical component",
                "importance": "Why it's critical",
                "implications": "What it means",
                "requirements": "What it requires",
                "timeline": "When it applies",
                "source": "Section reference"
            }
        ]
    }
}

CRITICAL REQUIREMENTS:
1. Provide actual quotes from the document wherever possible
2. Include section references for every significant point
3. Identify both explicit and implicit obligations
4. Note any missing standard provisions
5. Flag unusual or non-standard terms
6. Explain complex legal concepts in plain language
7. Highlight practical implications and real-world impacts
8. Include all monetary amounts, dates, and deadlines
9. Cross-reference related provisions
10. Note both rights and obligations of all parties
11. Identify potential issues and recommend solutions
12. Focus on specific details rather than generic statements
13. Use actual document text to support analysis
14. Highlight time-sensitive requirements
15. Note relationships between different provisions

Document to analyze (analyze thoroughly and provide specific details from the document):

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

        system_message = """You are an expert legal document analyzer with deep expertise in:
1. Document analysis and interpretation
2. Risk assessment and compliance
3. Legal implications and consequences
4. Practical implications and requirements
5. Professional obligations and ethics

Your task is to:
1. Thoroughly analyze the document
2. Extract and quote specific provisions
3. Identify all key terms, dates, and requirements
4. Note both explicit and implicit obligations
5. Flag any unusual or missing elements
6. Provide section-specific references
7. Explain in both legal and plain language
8. Focus on practical implications
9. Highlight all critical deadlines
10. Identify potential issues proactively

Your response must strictly follow the given JSON format. Any deviation will cause errors. Ensure that it is a valid JSON, and do not include anything else apart from the JSON structure."""

        response = client.chat.completions.create(
            model="gpt-4",
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

            # Ensure all necessary keys are present in the response, even if they are empty
            parsed_response.setdefault("document_metadata", {}).setdefault("type", {
                "primary": "",
                "category": "",
                "jurisdiction": ""
            })
            parsed_response.setdefault("document_metadata", {}).setdefault("matter_details", {
                "case_reference": "",
                "subject_matter": ""
            })
            parsed_response.setdefault("financial_structure", {}).setdefault("fee_arrangement", {
                "base_rates": []
            })
            parsed_response.setdefault("key_dates", {}).setdefault("document_dates", [])
            parsed_response.setdefault("critical_provisions", {}).setdefault("key_clauses", [])
            parsed_response.setdefault("risk_analysis", {
                "financial_risks": [],
                "legal_risks": [],
                "operational_risks": []
            })
            
            parsed_response["analysis_metadata"] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "document_length": len(extracted_text),
                "analysis_version": "3.0",
                "document_name": file.filename
            }

            return JSONResponse(content=parsed_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse analysis results: {str(e)}"
            )
            
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
        
        prompt = f"""Document text: {extracted_text}

Question: {question}

Provide a comprehensive answer that:
1. Quotes specific sections of the document
2. References exact page/section numbers
3. Explains in plain language
4. Notes any legal implications
5. Highlights practical impacts
6. Identifies any related provisions
7. Notes any important caveats
8. Suggests relevant follow-up considerations

Base your answer ONLY on the document's actual content."""
        
        response = client.chat.completions.create(
            model="gpt-4",
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

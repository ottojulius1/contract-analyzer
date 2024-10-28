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
    base_prompt = """As an expert legal document analyzer, provide a comprehensive analysis in valid JSON format. Your analysis MUST extract actual text and details from the document.

Required JSON Structure:
{
    "document_metadata": {
        "type": {
            "primary": "QUOTE exact document title/heading",
            "category": "QUOTE legal category (e.g., Family Law, Corporate Law)",
            "sub_type": "QUOTE specific type (e.g., Retainer Agreement, Employment Contract)"
        },
        "jurisdiction": {
            "state": "QUOTE state jurisdiction",
            "specific_courts": "QUOTE any specific courts mentioned",
            "governing_law": "QUOTE governing law provisions"
        },
        "parties": {
            "primary_parties": [
                {
                    "name": "QUOTE exact name",
                    "role": "QUOTE role/designation",
                    "type": "individual/entity",
                    "address": "QUOTE if provided",
                    "key_relationships": ["List any mentioned relationships"]
                }
            ],
            "related_parties": [
                {
                    "name": "QUOTE name",
                    "role": "QUOTE relationship to primary parties",
                    "relevance": "QUOTE context of involvement"
                }
            ],
            "hierarchy": {
                "representing_party": "QUOTE who represents whom",
                "opposing_parties": "QUOTE opposing parties",
                "third_parties": "QUOTE any third parties"
            }
        },
        "matter_details": {
            "case_reference": "QUOTE case/matter reference",
            "subject_matter": "QUOTE specific subject",
            "scope": {
                "included": ["LIST explicitly included items/services"],
                "excluded": ["LIST explicitly excluded items/services"],
                "conditional": ["LIST conditional services"]
            }
        }
    },
    "financial_structure": {
        "fee_arrangement": {
            "type": "QUOTE fee structure type",
            "base_rates": [
                {
                    "person": "QUOTE name/role",
                    "rate": "QUOTE exact rate",
                    "unit": "QUOTE billing unit",
                    "conditions": "QUOTE any conditions"
                }
            ],
            "retainers": [
                {
                    "amount": "QUOTE amount",
                    "type": "initial/replenishment/special",
                    "trigger": "QUOTE when required",
                    "terms": "QUOTE specific terms"
                }
            ],
            "additional_costs": [
                {
                    "type": "QUOTE cost type",
                    "amount": "QUOTE amount or rate",
                    "conditions": "QUOTE when applicable"
                }
            ]
        },
        "payment_terms": {
            "billing_cycle": "QUOTE billing frequency",
            "payment_deadline": "QUOTE payment terms",
            "late_payment_consequences": [
                {
                    "trigger": "QUOTE condition",
                    "consequence": "QUOTE result"
                }
            ],
            "interest_charges": {
                "rate": "QUOTE rate",
                "calculation_method": "QUOTE how calculated",
                "trigger_conditions": "QUOTE when applies"
            }
        }
    },
    "key_dates": {
        "document_dates": [
            {
                "date": "QUOTE exact date",
                "event": "QUOTE what happens",
                "significance": "HIGH/MEDIUM/LOW",
                "dependencies": ["List any dependent events"],
                "deadline_type": "firm/flexible/statutory"
            }
        ],
        "recurring_deadlines": [
            {
                "frequency": "QUOTE how often",
                "event": "QUOTE what happens",
                "conditions": "QUOTE any conditions",
                "importance": "HIGH/MEDIUM/LOW"
            }
        ],
        "conditional_dates": [
            {
                "trigger_event": "QUOTE what triggers",
                "deadline": "QUOTE timeframe",
                "required_action": "QUOTE what must be done",
                "consequences": "QUOTE what happens if missed"
            }
        ]
    },
    "critical_provisions": {
        "core_obligations": [
            {
                "party": "QUOTE who",
                "obligation": "QUOTE what must be done",
                "conditions": "QUOTE any conditions",
                "consequences": "QUOTE consequences of breach",
                "reference": "QUOTE section reference"
            }
        ],
        "key_clauses": [
            {
                "title": "QUOTE clause heading",
                "content": "QUOTE exact clause text",
                "significance": "QUOTE why important",
                "related_clauses": ["List related sections"],
                "reference": "QUOTE section reference"
            }
        ],
        "termination_provisions": [
            {
                "party": "QUOTE who can terminate",
                "grounds": "QUOTE valid reasons",
                "process": "QUOTE required steps",
                "consequences": "QUOTE what happens after",
                "notice_required": "QUOTE notice requirements"
            }
        ]
    },
    "risk_analysis": {
        "financial_risks": [
            {
                "risk": "QUOTE specific risk",
                "severity": "HIGH/MEDIUM/LOW",
                "likelihood": "HIGH/MEDIUM/LOW",
                "trigger_conditions": "QUOTE what triggers this risk",
                "mitigation_measures": "QUOTE any safeguards",
                "source_clause": "QUOTE relevant text"
            }
        ],
        "legal_risks": [
            {
                "risk": "QUOTE specific risk",
                "category": "procedural/substantive/compliance",
                "severity": "HIGH/MEDIUM/LOW",
                "implications": "QUOTE potential consequences",
                "mitigation_options": "QUOTE available remedies",
                "source_clause": "QUOTE relevant text"
            }
        ],
        "operational_risks": [
            {
                "risk": "QUOTE specific risk",
                "context": "QUOTE situation where it applies",
                "severity": "HIGH/MEDIUM/LOW",
                "preventive_measures": "QUOTE prevention steps",
                "source_clause": "QUOTE relevant text"
            }
        ]
    },
    "compliance_requirements": {
        "mandatory_actions": [
            {
                "requirement": "QUOTE what must be done",
                "deadline": "QUOTE when required",
                "responsible_party": "QUOTE who must do it",
                "verification_method": "QUOTE how verified",
                "consequences": "QUOTE consequences of non-compliance"
            }
        ],
        "prohibitions": [
            {
                "prohibited_action": "QUOTE what's prohibited",
                "scope": "QUOTE extent of prohibition",
                "exceptions": "QUOTE any exceptions",
                "consequences": "QUOTE consequences of violation"
            }
        ],
        "regulatory_requirements": [
            {
                "regulation": "QUOTE specific requirement",
                "applicable_authority": "QUOTE governing body",
                "compliance_steps": "QUOTE required actions",
                "verification": "QUOTE how compliance is verified"
            }
        ]
    },
    "cross_references": {
        "internal_references": [
            {
                "from_section": "QUOTE source section",
                "to_section": "QUOTE referenced section",
                "context": "QUOTE why referenced",
                "significance": "QUOTE importance of connection"
            }
        ],
        "external_references": [
            {
                "type": "statute/regulation/case law",
                "reference": "QUOTE specific reference",
                "context": "QUOTE how it applies",
                "significance": "QUOTE why important"
            }
        ]
    }
}

CRITICAL INSTRUCTIONS:
1. Every quoted field must contain ACTUAL TEXT from the document
2. Use "Not specified" only when information is truly absent
3. Include ALL monetary values found
4. Include ALL dates and deadlines
5. Include ALL named parties and their roles
6. Maintain relationships between connected provisions
7. Identify all risks, both explicit and implicit
8. Note any missing standard provisions
9. Highlight unusual or non-standard terms
10. Include specific section references whenever possible

Analyze this document and provide JSON matching this structure, populated with actual content from the document:

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

        system_message = """You are an expert legal document analyzer. Your task is to:
1. Extract specific information from legal documents
2. Quote relevant text directly
3. Identify key terms, dates, and provisions
4. Return analysis in valid JSON format
5. Never include placeholders or template text in response
6. Always use actual document content
7. Maintain proper JSON structure
8. Include all monetary values and dates found"""
        
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
            # Clean and parse JSON response
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            parsed_response = json.loads(cleaned_response)
            logger.debug("Successfully parsed JSON response")
            return JSONResponse(content=parsed_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse analysis results. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error during document analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing document: {str(e)}"
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

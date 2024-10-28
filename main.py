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
import re
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
def create_analysis_prompt(text: str) -> str:
    return """You are a highly experienced legal professional conducting a thorough document analysis. Think like both an expert lawyer and a client advocate. Your analysis must be comprehensive, detailed, and insightful.

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
            "governing_law": "Specific laws/regulations that apply"
        },
        "parties": [
            {
                "name": "Exact party name",
                "role": "Role in document",
                "primary_obligations": ["List key obligations"],
                "primary_rights": ["List key rights"],
                "key_restrictions": ["List main restrictions/limitations"]
            }
        ],
        "matter_info": {
            "subject": "Specific subject matter",
            "purpose": "Document's primary purpose",
            "scope": ["What's included", "What's excluded"],
            "special_circumstances": ["Any unique aspects"]
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
                "source": "Section reference"
            }
        ],
        "unusual_aspects": [
            {
                "aspect": "What's unusual",
                "why_significant": "Why it matters",
                "potential_impact": "Possible consequences"
            }
        ]
    },

    "key_terms_analysis": {
        "financial_terms": [
            {
                "term_type": "Type of financial term",
                "details": "Specific details",
                "amount": "Monetary value if applicable",
                "conditions": "Any conditions",
                "timing": "When it applies",
                "consequences": "What happens if not met",
                "source": "Section reference"
            }
        ],
        "operational_terms": [
            {
                "requirement": "What must be done",
                "who": "Responsible party",
                "when": "Timing/deadline",
                "how": "Process requirements",
                "consequences": "Results of non-compliance",
                "source": "Section reference"
            }
        ],
        "legal_terms": [
            {
                "term": "Legal requirement",
                "explanation": "Plain language explanation",
                "obligations": "What it requires",
                "implications": "Legal significance",
                "source": "Section reference"
            }
        ]
    },

    "critical_dates_deadlines": {
        "immediate_deadlines": [
            {
                "deadline": "Specific date/timeline",
                "requirement": "What's required",
                "responsible_party": "Who must act",
                "consequences": "What happens if missed",
                "source": "Section reference"
            }
        ],
        "recurring_obligations": [
            {
                "frequency": "How often",
                "requirement": "What's required",
                "details": "Specific requirements",
                "consequences": "Results of non-compliance"
            }
        ],
        "conditional_deadlines": [
            {
                "trigger": "What triggers the deadline",
                "timeline": "When it must be done",
                "requirement": "What must be done",
                "consequences": "What happens if missed"
            }
        ]
    },

    "rights_and_obligations": {
        "party_rights": [
            {
                "party": "Who has the right",
                "right": "Specific right",
                "conditions": "Conditions for exercising",
                "limitations": "Any limitations",
                "source": "Section reference"
            }
        ],
        "party_obligations": [
            {
                "party": "Who has obligation",
                "obligation": "Specific requirement",
                "standards": "Performance standards",
                "timing": "When it applies",
                "consequences": "Results of breach",
                "source": "Section reference"
            }
        ],
        "mutual_obligations": [
            {
                "obligation": "Shared requirement",
                "parties_involved": ["List parties"],
                "details": "Specific requirements",
                "source": "Section reference"
            }
        ]
    },

    "key_provisions_analysis": {
        "essential_clauses": [
            {
                "title": "Clause name",
                "content": "Exact quote",
                "plain_english": "Simple explanation",
                "legal_significance": "Legal meaning",
                "practical_implications": "Real-world impact",
                "related_provisions": ["Related sections"],
                "source": "Section reference"
            }
        ],
        "protective_clauses": [
            {
                "protection_type": "What it protects",
                "mechanism": "How it works",
                "beneficiary": "Who it protects",
                "limitations": "Any limitations",
                "source": "Section reference"
            }
        ],
        "procedural_requirements": [
            {
                "procedure": "What must be done",
                "steps": ["Specific steps required"],
                "timing": "When it applies",
                "importance": "Why it matters",
                "source": "Section reference"
            }
        ]
    },
    "comprehensive_risk_analysis": {
        "contractual_risks": [
            {
                "risk": "Specific risk",
                "scenario": "How it might occur",
                "likelihood": "HIGH/MEDIUM/LOW",
                "impact": "Potential consequences",
                "affected_party": "Who is at risk",
                "mitigation_options": "How to reduce risk",
                "source_provisions": ["Relevant sections"],
                "warning_signs": ["What to watch for"]
            }
        ],
        "compliance_risks": [
            {
                "requirement": "What's required",
                "risk": "Risk of non-compliance",
                "consequences": "Potential penalties/outcomes",
                "compliance_steps": "How to comply",
                "monitoring": "How to track compliance",
                "source": "Section reference"
            }
        ],
        "practical_risks": [
            {
                "risk": "Real-world risk",
                "context": "When it might occur",
                "warning_signs": ["What to watch for"],
                "preventive_steps": ["How to prevent"],
                "remedial_actions": ["What to do if it happens"],
                "source": "Section reference"
            }
        ],
        "relationship_risks": [
            {
                "risk": "Potential relationship issue",
                "context": "How it might arise",
                "early_signs": ["Warning indicators"],
                "prevention": "How to prevent",
                "management": "How to handle if it occurs",
                "source": "Section reference"
            }
        ]
    },

    "special_considerations": {
        "unique_features": [
            {
                "feature": "What's unique",
                "significance": "Why it matters",
                "implications": "What it means",
                "compare_to_standard": "How it differs from normal",
                "source": "Section reference"
            }
        ],
        "potential_issues": [
            {
                "issue": "Potential problem",
                "context": "When it might arise",
                "implications": "What it could mean",
                "preventive_measures": "How to prevent",
                "remedies": "How to address",
                "source": "Section reference"
            }
        ],
        "practice_tips": [
            {
                "tip": "Practical advice",
                "rationale": "Why it matters",
                "implementation": "How to follow",
                "benefits": "Why it helps",
                "source": "Section reference"
            }
        ]
    },

    "professional_obligations": {
        "attorney_obligations": [
            {
                "obligation": "Specific duty",
                "standard": "Performance standard",
                "scope": "What it covers",
                "limitations": "What it doesn't cover",
                "source": "Section reference"
            }
        ],
        "client_obligations": [
            {
                "obligation": "Required from client",
                "details": "Specific requirements",
                "timing": "When it applies",
                "importance": "Why it matters",
                "source": "Section reference"
            }
        ],
        "ethical_considerations": [
            {
                "issue": "Ethical concern",
                "rule_reference": "Governing rule",
                "requirements": "What's required",
                "best_practices": "How to handle",
                "source": "Section reference"
            }
        ]
    },

    "next_steps_and_recommendations": {
        "immediate_actions": [
            {
                "action": "What to do",
                "deadline": "When to do it",
                "responsibility": "Who does it",
                "details": "How to do it",
                "importance": "Why it's urgent"
            }
        ],
        "monitoring_requirements": [
            {
                "item": "What to monitor",
                "frequency": "How often",
                "method": "How to track",
                "indicators": "What to watch for",
                "response_plan": "What to do if issues arise"
            }
        ],
        "best_practices": [
            {
                "practice": "Recommended approach",
                "rationale": "Why recommended",
                "implementation": "How to implement",
                "benefits": "Expected benefits"
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
11. Focus on specific details rather than generic statements
12. Identify potential issues and recommend solutions
13. Use actual document text to support analysis
14. Highlight time-sensitive requirements
15. Note relationships between different provisions

Document to analyze (analyze thoroughly and provide specific details from the document):

"""
    return base_prompt + text
    @app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        # Extract text from PDF
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            extracted_text += f"\nPage {page_num + 1}:\n{page_text}"
        
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF")

        # System message for legal analysis
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
10. Identify potential issues proactively"""

        # Make API request
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better legal analysis
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": create_analysis_prompt(extracted_text)}
                ],
                temperature=0.1,  # Low temperature for more consistent, precise output
                max_tokens=4000,
                presence_penalty=0.1,  # Slight penalty to prevent repetition
                frequency_penalty=0.1
            )
            
            response_text = response.choices[0].message.content
            logger.debug("Received response from OpenAI")
            
            # Clean and parse JSON response
            try:
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                
                parsed_response = json.loads(cleaned_response)
                
                # Add metadata
                parsed_response["analysis_metadata"] = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "document_length": len(extracted_text),
                    "analysis_version": "3.0",
                    "document_name": file.filename
                }

                logger.info("Successfully processed and validated response")
                return JSONResponse(content=parsed_response)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Response text: {response_text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse analysis results: {str(e)}"
                )
                
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting analysis from AI: {str(api_error)}"
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
            model="gpt-4",  # Using GPT-4 for better comprehension
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

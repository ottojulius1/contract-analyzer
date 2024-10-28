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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_analysis_prompt(text: str) -> str:
    return """You are an expert legal analyst with comprehensive knowledge of all legal domains, jurisdictions, and document types. Provide a detailed, professional-grade analysis of the provided legal document following these requirements:

DOCUMENT CLASSIFICATION AND METADATA
1. Primary Classification:
   - Document type and sub-type
   - Jurisdiction and governing law
   - Filing/execution status
   - Document hierarchy/relationship
   - Authentication status

2. Executive Summary:
   - Opening statement of document purpose
   - Key provisions and material terms
   - Critical deadlines and timelines
   - Core obligations and rights
   - Essential risks and considerations
   - Immediate action items
   - Stakeholder implications

3. Legal Framework Analysis:
   - Governing law analysis
   - Jurisdictional requirements
   - Statutory framework
   - Regulatory compliance requirements
   - Industry-specific regulations
   - Required disclosures
   - Filing requirements

4. Comprehensive Risk Analysis:
   - Legal risks and exposure
   - Regulatory compliance risks
   - Business/operational risks
   - Financial implications
   - Procedural risks
   - Timeline/deadline risks
   - Stakeholder risks

5. Legal Elements Analysis:
   - Key provisions analysis
   - Rights and obligations
   - Conditions and prerequisites
   - Representations and warranties
   - Definitions and interpretations
   - Cross-references and dependencies
   - Precedent implications

6. Compliance Requirements:
   - Statutory compliance needs
   - Regulatory requirements
   - Industry standards
   - Filing obligations
   - Reporting requirements
   - Record-keeping obligations
   - Certification needs

7. Action Items and Timeline:
   - Immediate actions required
   - Upcoming deadlines
   - Filing requirements
   - Notice obligations
   - Response deadlines
   - Renewal/review dates
   - Compliance dates

8. Stakeholder Analysis:
   - Parties and roles
   - Rights and obligations
   - Authorization requirements
   - Notice requirements
   - Consent requirements
   - Approval chains

9. Document Integrity:
   - Execution requirements
   - Missing elements
   - Incomplete sections
   - Verification needs
   - Authentication requirements
   - Amendment status

10. Special Considerations:
    - Unique provisions
    - Non-standard elements
    - Industry-specific factors
    - Jurisdiction-specific requirements
    - Special conditions
    - Custom terms

Response Format:
{
    "document_metadata": {
        "primary_type": "detailed classification",
        "sub_type": "specific sub-category",
        "jurisdiction": "governing jurisdiction",
        "governing_law": "applicable law",
        "status": "current status",
        "hierarchy": {
            "document_level": "primary/subsidiary/amendment",
            "related_documents": ["list of related docs"],
            "dependencies": ["document dependencies"]
        },
        "authentication": {
            "requirements": ["required elements"],
            "status": "complete/incomplete",
            "missing_elements": ["missing items"]
        }
    },
    "executive_summary": {
        "purpose": "clear statement of document purpose",
        "key_points": [
            {
                "topic": "topic name",
                "details": "detailed explanation",
                "importance": "HIGH/MEDIUM/LOW"
            }
        ],
        "material_terms": [
            {
                "term": "term name",
                "explanation": "detailed explanation",
                "implications": "business/legal implications"
            }
        ],
        "critical_elements": [
            {
                "element": "element name",
                "details": "specific details",
                "urgency": "HIGH/MEDIUM/LOW"
            }
        ]
    },
    "legal_framework": {
        "governing_law_analysis": {
            "jurisdiction": "applicable jurisdiction",
            "key_statutes": ["relevant statutes"],
            "regulatory_framework": ["applicable regulations"],
            "precedent_cases": ["relevant cases"]
        },
        "compliance_requirements": [
            {
                "requirement": "specific requirement",
                "source": "statutory/regulatory source",
                "deadline": "compliance deadline",
                "status": "status of compliance"
            }
        ]
    },
    "risk_analysis": {
        "legal_risks": [
            {
                "risk": "detailed risk description",
                "severity": "HIGH/MEDIUM/LOW",
                "likelihood": "HIGH/MEDIUM/LOW",
                "impact": "detailed impact assessment",
                "mitigation": "mitigation strategies"
            }
        ],
        "compliance_risks": [
            {
                "risk": "compliance risk description",
                "regulation": "specific regulation",
                "consequences": "potential consequences",
                "mitigation": "compliance strategy"
            }
        ],
        "operational_risks": [
            {
                "risk": "operational risk description",
                "impact_areas": ["affected areas"],
                "mitigation": "operational strategies"
            }
        ]
    },
    "legal_elements": {
        "key_provisions": [
            {
                "provision": "provision name",
                "content": "provision content",
                "analysis": "detailed legal analysis",
                "implications": "legal/business implications",
                "requirements": "specific requirements",
                "references": ["related provisions"]
            }
        ],
        "definitions": [
            {
                "term": "defined term",
                "definition": "provided definition",
                "usage": "contextual usage",
                "implications": "legal implications"
            }
        ],
        "conditions": [
            {
                "condition": "condition description",
                "prerequisites": ["required elements"],
                "consequences": "if/then implications",
                "timeline": "temporal requirements"
            }
        ]
    },
    "action_items": {
        "immediate_actions": [
            {
                "action": "required action",
                "deadline": "specific deadline",
                "responsible_party": "who must act",
                "requirements": "what is required",
                "consequences": "consequences of inaction"
            }
        ],
        "upcoming_deadlines": [
            {
                "event": "deadline event",
                "date": "YYYY-MM-DD",
                "requirements": "what is required",
                "prior_actions": "preparatory steps"
            }
        ],
        "periodic_requirements": [
            {
                "requirement": "periodic task",
                "frequency": "how often",
                "next_due": "next due date",
                "responsibilities": "who is responsible"
            }
        ]
    },
    "stakeholder_analysis": {
        "parties": [
            {
                "party": "stakeholder name",
                "role": "role in document",
                "obligations": ["specific obligations"],
                "rights": ["specific rights"],
                "limitations": ["limitations on rights"]
            }
        ],
        "approval_requirements": [
            {
                "requirement": "approval description",
                "approver": "who must approve",
                "timeline": "when required",
                "process": "approval process"
            }
        ]
    },
    "document_integrity": {
        "execution_status": {
            "requirements": ["required elements"],
            "missing_elements": ["missing items"],
            "recommendations": ["recommended actions"]
        },
        "completeness": {
            "missing_sections": ["missing parts"],
            "incomplete_sections": ["incomplete parts"],
            "required_attachments": ["needed attachments"]
        }
    },
    "special_considerations": {
        "unique_elements": [
            {
                "element": "unique feature",
                "analysis": "detailed analysis",
                "implications": "specific implications"
            }
        ],
        "jurisdiction_specific": [
            {
                "requirement": "specific requirement",
                "jurisdiction": "applicable jurisdiction",
                "compliance_needs": "what is required"
            }
        ],
        "industry_specific": [
            {
                "requirement": "industry requirement",
                "industry": "specific industry",
                "compliance_needs": "what is required"
            }
        ]
    }
}

Document text to analyze:
{text}

Provide a comprehensive analysis that:
1. Is specific to the exact document type
2. Contains detailed, actionable insights
3. Highlights all critical elements
4. Identifies potential issues and risks
5. Provides clear next steps
6. References specific document sections
7. Notes any missing or incomplete elements
8. Includes jurisdiction-specific considerations"""

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        # Read and extract text from PDF
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        extracted_text = ""
        
        logger.debug(f"Starting text extraction from {file.filename}")
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            extracted_text += f"\nPage {page_num + 1}:\n{page_text}"
        
        logger.debug(f"Extracted {len(extracted_text)} characters from PDF")
        
        # Create analysis prompt
        prompt = create_analysis_prompt(extracted_text)
        
        # Get OpenAI response with enhanced parameters
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for more sophisticated legal analysis
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst with comprehensive knowledge of all legal domains, jurisdictions, and document types. Provide detailed, professional-grade analysis with specific citations to document sections."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent, precise responses
            max_tokens=4000,  # Increased for comprehensive analysis
            presence_penalty=0.1,  # Slight penalty to prevent repetition
            frequency_penalty=0.1  # Slight penalty to encourage diverse language
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
            extracted_text += f"\nPage {page_num + 1}:\n{pdf_reader.pages[page_num].extract_text()}"
        
        prompt = f"""Document Context: {extracted_text}

Question: {question}

Provide a comprehensive answer that:
1. Cites specific sections of the document
2. Explains legal implications
3. Identifies relevant context
4. Notes any uncertainties or ambiguities
5. References related provisions
6. Suggests follow-up considerations

Focus on providing accurate, legally relevant information based solely on the document content."""
        
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for more accurate legal analysis
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal analyst providing detailed, accurate answers about legal documents. Cite specific sections and explain legal implications."
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

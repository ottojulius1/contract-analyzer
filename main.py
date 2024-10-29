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
    },
    "key_terms_analysis": {
        "financial_terms": [
            {
                "term_type": "Type of financial term",
                "details": "Specific details",
                "amount": "Monetary value if applicable",
                "conditions": "Any conditions",
                "timing": "When it applies",
                "calculation_method": "How it's calculated",
                "payment_terms": "Payment requirements",
                "consequences": "What happens if not met",
                "exceptions": "Any exceptions",
                "source": "Section reference",
                "related_provisions": ["Related sections"]
            }
        ],
        "operational_terms": [
            {
                "requirement": "What must be done",
                "who": "Responsible party",
                "when": "Timing/deadline",
                "how": "Process requirements",
                "standards": "Performance standards",
                "verification": "How compliance is verified",
                "reporting": "Reporting requirements",
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
                "applicable_law": "Governing law/regulation",
                "compliance_requirements": "How to comply",
                "exceptions": "Any exceptions",
                "precedents": "Relevant legal precedents",
                "source": "Section reference"
            }
        ],
        "defined_terms": [
            {
                "term": "Defined term",
                "definition": "Exact definition from document",
                "context": "How it's used",
                "significance": "Why it matters",
                "related_terms": ["Related definitions"],
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
                "prerequisites": "What must happen first",
                "consequences": "What happens if missed",
                "extensions": "Possible extensions",
                "notification_requirements": "Who must be notified",
                "documentation": "Required documentation",
                "source": "Section reference"
            }
        ],
        "recurring_obligations": [
            {
                "frequency": "How often",
                "requirement": "What's required",
                "details": "Specific requirements",
                "timing": "When within period",
                "responsible_party": "Who must perform",
                "tracking_method": "How to track",
                "verification": "How to verify completion",
                "consequences": "Results of non-compliance"
            }
        ],
        "conditional_deadlines": [
            {
                "trigger": "What triggers the deadline",
                "timeline": "When it must be done",
                "requirement": "What must be done",
                "responsible_party": "Who must act",
                "notification": "Who must be notified",
                "documentation": "Required documentation",
                "consequences": "What happens if missed"
            }
        ],
        "key_dates": [
            {
                "date": "Specific date",
                "event": "What happens",
                "significance": "Why it matters",
                "requirements": "What's required",
                "parties_involved": ["Who's involved"],
                "source": "Section reference"
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
                "exercise_procedure": "How to exercise",
                "notice_requirements": "Required notifications",
                "time_constraints": "Time limits",
                "exclusions": "What's not included",
                "source": "Section reference"
            }
        ],
        "party_obligations": [
            {
                "party": "Who has obligation",
                "obligation": "Specific requirement",
                "standards": "Performance standards",
                "timing": "When it applies",
                "prerequisites": "What must happen first",
                "compliance_method": "How to comply",
                "verification": "How compliance is verified",
                "consequences": "Results of breach",
                "cure_provisions": "How to fix breaches",
                "source": "Section reference"
            }
        ],
        "mutual_obligations": [
            {
                "obligation": "Shared requirement",
                "parties_involved": ["List parties"],
                "details": "Specific requirements",
                "coordination": "How parties work together",
                "responsibilities": "Individual responsibilities",
                "dispute_resolution": "How to resolve disagreements",
                "source": "Section reference"
            }
        ],
        "conditional_obligations": [
            {
                "trigger": "What activates obligation",
                "obligation": "What must be done",
                "responsible_party": "Who must do it",
                "timeline": "When it must be done",
                "conditions": "Required conditions",
                "verification": "How to verify",
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
                "requirements": "What it requires",
                "enforcement": "How it's enforced",
                "relationship": "Connection to other clauses",
                "precedent_analysis": "Relevant legal precedents",
                "source": "Section reference"
            }
        ],
        "protective_clauses": [
            {
                "protection_type": "What it protects",
                "mechanism": "How it works",
                "beneficiary": "Who it protects",
                "scope": "What's covered",
                "limitations": "What's not covered",
                "enforcement": "How it's enforced",
                "duration": "How long it lasts",
                "exceptions": "When it doesn't apply",
                "source": "Section reference"
            }
        ],
        "procedural_requirements": [
            {
                "procedure": "What must be done",
                "steps": ["Specific steps required"],
                "timing": "When it applies",
                "parties_involved": ["Who's involved"],
                "documentation": "Required documentation",
                "verification": "How to verify completion",
                "consequences": "What happens if not followed",
                "source": "Section reference"
            }
        ],
        "termination_provisions": [
            {
                "scenario": "Termination situation",
                "requirements": "What's required",
                "process": "Steps to follow",
                "notice": "Notice requirements",
                "cure_rights": "Rights to fix issues",
                "consequences": "Results of termination",
                "surviving_obligations": "What continues after",
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
                "trigger_events": ["What could cause this"],
                "early_warning_signs": ["Signs to watch for"],
                "mitigation_options": "How to reduce risk",
                "contingency_plans": "What to do if it happens",
                "source_provisions": ["Relevant sections"],
                "monitoring_requirements": "How to monitor",
                "insurance_requirements": "Required coverage"
            }
        ],
        "compliance_risks": [
            {
                "requirement": "What's required",
                "risk": "Risk of non-compliance",
                "regulatory_framework": "Governing regulations",
                "consequences": "Potential penalties/outcomes",
                "likelihood": "HIGH/MEDIUM/LOW",
                "impact": "Severity of consequences",
                "compliance_steps": "How to comply",
                "monitoring": "How to track compliance",
                "reporting": "Required reporting",
                "documentation": "Required documentation",
                "source": "Section reference"
            }
        ],
        "practical_risks": [
            {
                "risk": "Real-world risk",
                "context": "When it might occur",
                "warning_signs": ["What to watch for"],
                "business_impact": "Effect on operations",
                "financial_impact": "Cost implications",
                "reputation_impact": "Effect on reputation",
                "preventive_steps": ["How to prevent"],
                "remedial_actions": ["What to do if it happens"],
                "insurance": "Available coverage",
                "source": "Section reference"
            }
        ],
        "relationship_risks": [
            {
                "risk": "Potential relationship issue",
                "context": "How it might arise",
                "parties_affected": ["Who's affected"],
                "early_signs": ["Warning indicators"],
                "impact_on_performance": "Effect on obligations",
                "communication_requirements": "Required communications",
                "prevention": "How to prevent",
                "management": "How to handle if it occurs",
                "escalation_process": "How to escalate issues",
                "resolution_methods": "How to resolve",
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
                "advantages": ["Benefits"],
                "disadvantages": ["Drawbacks"],
                "special_requirements": "Special handling needed",
                "source": "Section reference"
            }
        ],
        "potential_issues": [
            {
                "issue": "Potential problem",
                "context": "When it might arise",
                "implications": "What it could mean",
                "early_indicators": ["Warning signs"],
                "preventive_measures": "How to prevent",
                "monitoring": "How to track",
                "remedies": "How to address",
                "escalation": "When to escalate",
                "source": "Section reference"
            }
        ],
        "practice_tips": [
            {
                "tip": "Practical advice",
                "rationale": "Why it matters",
                "implementation": "How to follow",
                "benefits": "Why it helps",
                "timing": "When to apply",
                "resources_needed": "What's required",
                "success_metrics": "How to measure success",
                "source": "Section reference"
            }
        ],
        "industry_specific": [
            {
                "consideration": "Industry-specific issue",
                "relevance": "Why it matters",
                "industry_standards": "Applicable standards",
                "best_practices": "Industry best practices",
                "regulatory_aspects": "Special regulations",
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
                "compliance_requirements": "How to comply",
                "ethical_considerations": "Ethical aspects",
                "documentation_needed": "Required records",
                "source": "Section reference"
            }
        ],
        "client_obligations": [
            {
                "obligation": "Required from client",
                "details": "Specific requirements",
                "timing": "When it applies",
                "importance": "Why it matters",
                "consequences": "If not fulfilled",
                "support_needed": "Help required",
                "verification": "How to verify",
                "source": "Section reference"
            }
        ],
        "ethical_considerations": [
            {
                "issue": "Ethical concern",
                "rule_reference": "Governing rule",
                "requirements": "What's required",
                "limitations": "What's prohibited",
                "best_practices": "How to handle",
                "documentation": "Required records",
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
                "importance": "Why it's urgent",
                "prerequisites": "What's needed first",
                "verification": "How to confirm completion",
                "documentation": "Required records"
            }
        ],
        "monitoring_requirements": [
            {
                "item": "What to monitor",
                "frequency": "How often",
                "method": "How to track",
                "responsible_party": "Who monitors",
                "indicators": "What to watch for",
                "reporting": "How to report",
                "response_plan": "What to do if issues arise",
                "documentation": "Required records"
            }
        ],
        "best_practices": [
            {
                "practice": "Recommended approach",
                "rationale": "Why recommended",
                "implementation": "How to implement",
                "timing": "When to implement",
                "resources": "What's needed",
                "benefits": "Expected benefits",
                "measures": "How to measure success"
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
10. Identify potential issues proactively"""

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
            if cleaned_response.startswith("
json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("
"):
                cleaned_response = cleaned_response[:-3]
            
            parsed_response = json.loads(cleaned_response)
            
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

"""
WDC Labs AI Backend
FastAPI application for the immersive virtual office AI system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import os
import httpx
import io
import requests
import mimetypes
from PIL import Image

from .schemas import (
    ChatRequest, ChatResponse,
    BioAssessmentRequest, BioAssessmentResponse,
    SubmissionReviewRequest, SubmissionReviewResponse,
    PortfolioBulletRequest, PortfolioBulletResponse,
    OnboardingIntroRequest, OnboardingIntroResponse, OnboardingIntroMessage, AgentName
)
import re
import json
from .orchestrator import Orchestrator

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize orchestrator
orchestrator = Orchestrator(model)

# Create FastAPI app
app = FastAPI(
    title="WDC Labs AI Backend",
    description="Immersive Virtual Office AI System with Multi-Agent Architecture",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "WDC Labs AI Backend is running"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model": "gemini-2.5-flash",
        "agents": ["Tolu", "Emem", "Sola", "Kemi"]
    }


# ============ CHAT ENDPOINTS ============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes messages to appropriate agent.
    
    The orchestrator determines which agent should respond based on:
    - Message content (keywords)
    - Context (is_submission, is_first_login, etc.)
    """
    try:
        response = await orchestrator.route_message(
            message=request.message,
            context=request.context,
            chat_history=request.chat_history or []
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ ONBOARDING ENDPOINTS ============

@app.post("/assess-bio", response_model=BioAssessmentResponse)
async def assess_bio(request: BioAssessmentRequest):
    """
    Tolu assesses the user's bio/resume and assigns a skill level.
    
    Returns:
    - Level 0 (Foundation): No computer skills, digital literacy focus
    - Level 1 (Beginner): Some education, no real experience
    - Level 2 (Intermediate): Has degree or specific technical skills
    """
    try:
        bio_text = request.bio_text or ""
        cv_text = ""
        
        # Download and parse CV if URL provided
        cv_url = request.cv_url or request.file_url
        if cv_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(cv_url)
                    if response.status_code == 200:
                        # Determine file type from URL or content
                        is_pdf = cv_url.lower().endswith('.pdf') or response.headers.get('content-type', '').startswith('application/pdf')
                        is_docx = cv_url.lower().endswith('.docx') or 'openxmlformats' in response.headers.get('content-type', '')
                        is_doc = cv_url.lower().endswith('.doc')
                        
                        if is_pdf:
                            # Parse PDF
                            try:
                                import PyPDF2
                                pdf_file = io.BytesIO(response.content)
                                pdf_reader = PyPDF2.PdfReader(pdf_file)
                                for page in pdf_reader.pages:
                                    cv_text += page.extract_text() or ""
                                print(f"Extracted {len(cv_text)} characters from PDF")
                            except Exception as pdf_err:
                                print(f"PDF parsing error: {pdf_err}")
                        
                        elif is_docx:
                            # Parse DOCX
                            try:
                                from docx import Document
                                docx_file = io.BytesIO(response.content)
                                doc = Document(docx_file)
                                for para in doc.paragraphs:
                                    cv_text += para.text + "\n"
                                print(f"Extracted {len(cv_text)} characters from DOCX")
                            except Exception as docx_err:
                                print(f"DOCX parsing error: {docx_err}")
                        
                        elif is_doc:
                            # DOC files need different handling (antiword or similar)
                            print("DOC format detected - limited support, treating as binary")
                            cv_text = "[DOC file uploaded - text extraction limited]"
                        
                        else:
                            # Try as plain text
                            cv_text = response.text[:5000] if response.text else ""
                            print(f"Treating as plain text: {len(cv_text)} characters")
                            
            except Exception as download_err:
                print(f"CV download error: {download_err}")
        
        # Accept bio_text, file_url, or cv_url
        if not bio_text and not cv_text and not request.cv_url:
            raise HTTPException(
                status_code=400, 
                detail="Either bio_text, file_url, or cv_url must be provided"
            )
        
        # Combine bio and CV text for assessment
        assessment_text = ""
        if bio_text:
            assessment_text = bio_text
        if cv_text:
            assessment_text += f"\n\n[CV Content]:\n{cv_text[:3000]}"  # Limit CV text
        elif cv_url and not cv_text:
            assessment_text += "\n[User uploaded a CV file]"
        
        # If still empty, use a fallback
        if not assessment_text.strip():
            assessment_text = "New user with uploaded CV, no extracted text available."
        
        result = await orchestrator.assess_bio(assessment_text, request.track)
        
        return BioAssessmentResponse(
            response_text=result.get("response_text", "Welcome to WDC Labs."),
            assessed_level=result.get("assessed_level", "Level 1"),
            reasoning=result.get("reasoning", "Assessment completed."),
            warmup_mode=result.get("warmup_mode", False)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"assess_bio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ SUBMISSION ENDPOINTS ============

@app.post("/review-submission", response_model=SubmissionReviewResponse)
async def review_submission(request: SubmissionReviewRequest):
    """
    Sola reviews a user's work submission.
    
    Implements the 60% Rejection Rule - only truly excellent work passes.
    If passed, Kemi automatically generates a CV bullet point.
    """
    try:
        submission_content = request.file_content or f"[File submitted: {request.file_url}]"
        
        result = await orchestrator.review_submission(
            task_title=request.task_title,
            task_brief=request.task_brief,
            submission_content=submission_content,
            client_constraints=None
        )
        
        portfolio_bullet = None
        if result.get("passed") and result.get("portfolio_bullet"):
            portfolio_bullet = result["portfolio_bullet"].get("bullet_point")
        
        return SubmissionReviewResponse(
            feedback=result.get("feedback", "Review completed."),
            passed=result.get("passed", False),
            score=result.get("score"),
            portfolio_bullet=portfolio_bullet
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interrogate-submission")
async def interrogate_submission(submission_content: str, approach: str):
    """
    Sola's Socratic Defense - interrogate a user about their choices.
    
    Used to detect copied/AI-generated work by asking about specific decisions.
    """
    try:
        questions = await orchestrator.interrogate_submission(
            submission_content=submission_content,
            approach_used=approach
        )
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ PORTFOLIO ENDPOINTS ============

@app.post("/translate-to-cv", response_model=PortfolioBulletResponse)
async def translate_to_cv(request: PortfolioBulletRequest):
    """
    Kemi translates a completed task into a CV bullet point.
    """
    try:
        from .agents import kemi
        result = await kemi.translate_to_cv_bullet(
            task_title=request.task_title,
            task_description=request.task_description,
            user_accomplishment=request.user_submission,
            model=model
        )
        return PortfolioBulletResponse(
            skill_tag=result.get("skill_tag", "General"),
            bullet_point=result.get("bullet_point", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ INTERRUPTION ENDPOINTS ============

@app.post("/generate-interruption")
async def generate_interruption(current_task: str, interruption_type: str = "scope_change"):
    """
    Generate a mid-task client interruption (The "Moving Target").
    
    Types: scope_change, constraint_added, urgent_pivot, data_correction
    """
    try:
        interruption = await orchestrator.generate_client_interruption(
            current_task=current_task,
            interruption_type=interruption_type
        )
        return {"agent": "Emem", "message": interruption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ COACHING ENDPOINTS ============

@app.post("/soft-skills-feedback")
async def get_soft_skills_feedback(recent_interactions: list):
    """
    Kemi analyzes recent interactions and provides soft skills feedback.
    """
    try:
        feedback = await orchestrator.get_soft_skills_feedback(recent_interactions)
        return {"agent": "Kemi", "feedback": feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mock-interview")
async def mock_interview(
    interview_type: str = "behavioral",
    question_number: int = 1,
    previous_answer: str = None
):
    """
    Conduct a mock interview session with Kemi.
    
    Types: behavioral, technical, situational
    """
    try:
        result = await orchestrator.conduct_mock_interview(
            interview_type=interview_type,
            question_number=question_number,
            previous_answer=previous_answer
        )
        return {"agent": "Kemi", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ ONBOARDING ENDPOINTS ============

@app.post("/onboarding-intro", response_model=OnboardingIntroResponse)
async def generate_onboarding_intro(request: OnboardingIntroRequest):
    """
    Generate personalized team introduction messages for onboarding.
    Returns a sequence of messages from each agent with typing delays.
    """
    try:
        # Build context for AI
        bio_context = f"Based on their background: {request.bio_summary}" if request.bio_summary else ""
        level_context = f" (assessed as {request.user_level})" if request.user_level else ""
        
        # Generate personalized messages using AI
        prompt = f"""
You are generating a team onboarding introduction for a virtual office internship.

New Intern Details:
- Name: {request.user_name}
- Track: {request.track}{level_context}
{bio_context}

Generate exactly 7 short, personalized messages that the team members would say to welcome this intern.
Each message should be 1-3 sentences max, professional but warm.
Use their name naturally where appropriate.

The messages should be in this EXACT order and format:
1. Tolu (HR): Announces patching in the team, sets professional tone
2. Tolu (HR): Introduces the intern to the team with their name and track
3. Kemi (Career Coach): Warm welcome, acknowledges their background, explains portfolio building
4. Kemi (Career Coach): Explains how she translates their work into career outcomes
5. Emem (Project Manager): Brief, deadline-focused, mentions first task coming soon
6. Sola (Technical Reviewer): Professional, explains review standards and expectations
7. Tolu (HR): Asks if intern has questions before signing off

Respond in this JSON format:
{{
  "messages": [
    {{"agent": "Tolu", "message": "..."}},
    {{"agent": "Tolu", "message": "..."}},
    {{"agent": "Kemi", "message": "..."}},
    {{"agent": "Kemi", "message": "..."}},
    {{"agent": "Emem", "message": "..."}},
    {{"agent": "Sola", "message": "..."}},
    {{"agent": "Tolu", "message": "..."}}
  ]
}}
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Parse JSON from response
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            messages = []
            cumulative_delay = 0
            
            # Track length of previous message to add reading time
            prev_msg_length = 0
            
            for i, msg in enumerate(data.get("messages", [])):
                # Calculate typing delay based on message length (approx 5 chars per word, 40 WPM = 200 CPM)
                # 300ms per char is too slow, let's do ~50-80ms per char + base offset
                msg_length = len(msg.get("message", ""))
                
                # Base typing time: 1.5s minimum, plus 60ms per character
                typing_time = max(2000, 1500 + (msg_length * 60))
                
                # Add reading pause from previous message (fast readers approx 20ms/char)
                reading_pause = max(500, prev_msg_length * 30) if i > 0 else 0
                
                # Update cumulative delay
                cumulative_delay += reading_pause + typing_time
                
                prev_msg_length = msg_length
                
                agent_name = msg.get("agent", "Tolu")
                messages.append(OnboardingIntroMessage(
                    agent=AgentName(agent_name),
                    message=msg.get("message", ""),
                    typing_delay_ms=cumulative_delay
                ))
            
            return OnboardingIntroResponse(messages=messages)
        else:
            raise ValueError("Could not parse AI response")
            
    except Exception as e:
        # Fallback to default messages
        print(f"Error generating onboarding intro: {e}")
        default_messages = [
            OnboardingIntroMessage(agent=AgentName.TOLU, message="Alright, let me patch in the team. These are the people who will determine if you get a recommendation letter or not.", typing_delay_ms=2000),
            OnboardingIntroMessage(agent=AgentName.TOLU, message=f"Team, this is the new intern, {request.user_name}. Assigned to the {request.track} unit.", typing_delay_ms=4500),
            OnboardingIntroMessage(agent=AgentName.KEMI, message=f"Hi {request.user_name}! I'm Kemi, your career coach. I'll be translating your work here into a portfolio that gets you hired.", typing_delay_ms=7500),
            OnboardingIntroMessage(agent=AgentName.KEMI, message="You do the work, I'll build the career. Even if you're starting from zero, in 12 months, you'll look like a pro on paper.", typing_delay_ms=10500),
            OnboardingIntroMessage(agent=AgentName.EMEM, message=f"Welcome {request.user_name}. I don't care about your background, I care about deadlines. Your first brief is coming in 5 minutes.", typing_delay_ms=13500),
            OnboardingIntroMessage(agent=AgentName.SOLA, message=f"Hi {request.user_name}. I'm Sola. I review all technical output. I reject about 60% of first drafts. Don't take it personally, just fix it.", typing_delay_ms=16500),
            OnboardingIntroMessage(agent=AgentName.TOLU, message=f"{request.user_name}, any questions before I sign off?", typing_delay_ms=19000),
        ]
        return OnboardingIntroResponse(messages=default_messages)








# ============ WORK SUBMISSION REVIEW ============

@app.post("/review-submission", response_model=SubmissionReviewResponse)
async def review_submission(request: SubmissionReviewRequest):
    """Sola reviews work submission with real file analysis."""
    # Construct context from chat history
    chat_context = ""
    if request.chat_history:
        chat_context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in request.chat_history])
    
    prompt = f"""
    Role: You are SOLA, the Lead Technical Supervisor at WDC Labs.
    Personality: Professional, strict, high standards. You do not tolerate laziness.
    Task: Review this intern's work submission.
    
    Context:
    Task Title: {request.task_title}
    Task Brief: {request.task_brief}
    Recent Chat Context: {chat_context}
    
    Submission:
    File URL: {request.file_url or 'No file attached'}
    User Notes: {request.file_content or 'No content provided'}
    
    Review Logic:
    1. AUTOMATIC REJECT: If there is NO file AND notes are empty/placeholder, REJECT immediately.
    2. CHECK FILE: If a file is attached, analyze its contents thoroughly against the Brief.
    3. 60% RULE: Enforce a 60% rejection rate for first attempts unless exceptional.
    4. FEEDBACK: Be stern but constructive.
    
    Output JSON ONLY:
    {{
        "feedback": "string",
        "passed": boolean,
        "score": integer (0-100),
        "portfolio_bullet": "string or null"
    }}
    """
    
    try:
        content = [prompt]
        
        # Download and append file content if URL exists
        if request.file_url and request.file_url.startswith('http'):
            try:
                print(f"Downloading submission: {request.file_url}")
                res = requests.get(request.file_url)
                if res.status_code == 200:
                    file_data = res.content
                    
                    # Guess Type
                    # Use provided filename or url path
                    filename = request.file_url.split('/')[-1]
                    mime_type, _ = mimetypes.guess_type(filename)
                    
                    if mime_type and mime_type.startswith('image'):
                        image = Image.open(io.BytesIO(file_data))
                        content.append(image)
                        print("Attached Image to prompt")
                    elif mime_type == 'application/pdf':
                        content.append({
                            "mime_type": "application/pdf",
                            "data": file_data
                        })
                        print("Attached PDF to prompt")
                    else:
                        # Fallback text
                        try:
                            text = file_data.decode('utf-8')
                             # Append as text
                            content.append(f"\n[Attached File Content]:\n{text}")
                        except:
                            print("Could not decode file text")
            except Exception as e:
                print(f"File download error: {e}")

        # Call Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(content)
        response_text = response.text
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return SubmissionReviewResponse(
                agent="Sola",
                feedback=data.get("feedback", "Submission received."),
                passed=data.get("passed", False),
                score=data.get("score", 0),
                portfolio_bullet=data.get("portfolio_bullet")
            )
        else:
             return SubmissionReviewResponse(
                agent="Sola",
                feedback="I couldn't parse your submission. Please check formatting.",
                passed=False,
                score=0
            )
            
    except Exception as e:
        print(f"Error reviewing submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ STARTUP EVENT ============

@app.on_event("startup")
async def startup_event():
    """Log startup."""
    print("🚀 WDC Labs AI Backend starting...")
    print("✅ Gemini AI configured")
    print("✅ Agents: Tolu, Emem, Sola, Kemi ready")
    print("✅ Orchestrator initialized")

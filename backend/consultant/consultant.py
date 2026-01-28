"""
DigitalOcean Agent Consultant
Sends user intake data, roadmaps, and chat messages to DigitalOcean Agent for personalized advice.
Falls back to Gemini API if DigitalOcean Agent is unavailable.
"""

import os
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Token limits for context management
MAX_CONTEXT_TOKENS = 8000  # Conservative limit to avoid hitting API limits
APPROX_CHARS_PER_TOKEN = 4  # Rough estimate: 1 token ≈ 4 characters


class ConsultantError(Exception):
    """Custom exception for consultant errors"""
    pass


def format_context(intake_data: Optional[Dict[str, Any]], programs: Optional[List[Dict[str, Any]]], max_programs: int = 3) -> str:
    """
    Format intake data and programs into a structured context string.
    
    Args:
        intake_data: Dictionary with student intake information
        programs: List of program dictionaries with roadmap data
        max_programs: Maximum number of programs to include (default 3 to manage token limits)
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # Format intake data
    if intake_data:
        context_parts.append("=== STUDENT PROFILE ===")
        context_parts.append(f"Name: {intake_data.get('name', 'N/A')}")
        context_parts.append(f"Email: {intake_data.get('email', 'N/A')}")
        context_parts.append(f"Grade Level: {intake_data.get('grade', 'N/A')}")
        context_parts.append(f"Wants Co-op: {'Yes' if intake_data.get('wants_coop') else 'No'}")
        
        # Interests
        if intake_data.get('interests'):
            context_parts.append(f"\nInterests: {', '.join(intake_data['interests'])}")
        
        # Extracurriculars (limit to top 5)
        if intake_data.get('extra_curriculars'):
            context_parts.append("\nExtracurricular Activities:")
            for ec in intake_data['extra_curriculars'][:5]:
                name = ec.get('name', 'N/A')
                level = ec.get('leadership_level', 'N/A')
                context_parts.append(f"  • {name} (Leadership Level: {level}/4)")
            if len(intake_data['extra_curriculars']) > 5:
                context_parts.append(f"  ... and {len(intake_data['extra_curriculars']) - 5} more")
        
        # Courses (limit to recent/important ones)
        if intake_data.get('courses_taken'):
            context_parts.append("\nCourses Taken:")
            for course in intake_data['courses_taken'][:8]:
                course_code = course.get('course', 'N/A')
                grade = course.get('grade', 'N/A')
                context_parts.append(f"  • {course_code}: {grade}%")
            if len(intake_data['courses_taken']) > 8:
                context_parts.append(f"  ... and {len(intake_data['courses_taken']) - 8} more")
    
    # Format programs/roadmaps (limit to max_programs)
    if programs:
        context_parts.append("\n\n=== APPLICATION ROADMAPS ===")
        
        for i, program in enumerate(programs[:max_programs], 1):
            university = program.get('universityName', 'N/A')
            program_name = program.get('programName', 'N/A')
            deadline = program.get('deadline', 'N/A')
            progress = program.get('overallProgress', 0)
            
            context_parts.append(f"\n{i}. {university} - {program_name}")
            context_parts.append(f"   Application Deadline: {deadline}")
            context_parts.append(f"   Overall Progress: {progress}%")
            
            # Application steps (show only incomplete or in-progress)
            steps = program.get('steps', [])
            if steps:
                incomplete_steps = [s for s in steps if s.get('status') != 'complete']
                completed = sum(1 for s in steps if s.get('status') == 'complete')
                
                context_parts.append(f"   Tasks: {completed}/{len(steps)} completed")
                
                if incomplete_steps:
                    context_parts.append("   Pending Tasks:")
                    for step in incomplete_steps[:5]:  # Show max 5 pending tasks
                        status = step.get('status', 'todo')
                        title = step.get('title', 'N/A')
                        due_date = step.get('dueDate', 'No deadline')
                        emoji = '→' if status == 'in-progress' else '○'
                        context_parts.append(f"     [{emoji}] {title} (Due: {due_date})")
            
            # Bonus tasks (show count only)
            bonus_tasks = program.get('bonusTasks', [])
            if bonus_tasks:
                completed_bonus = sum(1 for t in bonus_tasks if t.get('status') == 'complete')
                context_parts.append(f"   Bonus Tasks: {completed_bonus}/{len(bonus_tasks)} completed")
        
        if len(programs) > max_programs:
            context_parts.append(f"\n... and {len(programs) - max_programs} more programs")
    
    return "\n".join(context_parts)


def _estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text: Text to estimate tokens for
    
    Returns:
        Estimated token count
    """
    return len(text) // APPROX_CHARS_PER_TOKEN


def _truncate_context_if_needed(context: str, message: str, system_prompt: str) -> str:
    """
    Truncate context if total tokens exceed limit.
    
    Args:
        context: Formatted context string
        message: User message
        system_prompt: System prompt
    
    Returns:
        Potentially truncated context
    """
    total_text = system_prompt + context + message
    estimated_tokens = _estimate_tokens(total_text)
    
    if estimated_tokens > MAX_CONTEXT_TOKENS:
        # Calculate how much to truncate
        excess_tokens = estimated_tokens - MAX_CONTEXT_TOKENS
        excess_chars = excess_tokens * APPROX_CHARS_PER_TOKEN
        
        # Truncate context from the end (keep most recent/important info at top)
        if len(context) > excess_chars:
            context = context[:-excess_chars] + "\n\n[Context truncated due to length]"
        else:
            # If context is too large, keep only essential info
            context = "[Context truncated - too large to include full details]"
    
    return context


def _try_gemini_fallback(message: str, intake_data: Optional[Dict[str, Any]], programs: Optional[List[Dict[str, Any]]]) -> str:
    """
    Fallback to Gemini API if DigitalOcean Agent fails.
    
    Args:
        message: User's question
        intake_data: Optional student intake data
        programs: Optional program roadmaps
    
    Returns:
        Response from Gemini API
    
    Raises:
        ConsultantError: If Gemini also fails
    """
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConsultantError("GEMINI_API_KEY not configured - cannot use fallback")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Format context
        context = format_context(intake_data, programs, max_programs=2)  # Limit to 2 programs for Gemini
        
        if context:
            prompt = f"""You are an Ontario university admissions consultant. Here is the student's information:

{context}

Provide helpful, personalized advice based on their profile.

Student Question: {message}"""
        else:
            prompt = f"""You are an Ontario university admissions consultant. Provide helpful advice about university applications and programs.

Student Question: {message}"""
        
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise ConsultantError("Gemini returned empty response")
        
        return response.text
        
    except ImportError:
        raise ConsultantError("Gemini fallback unavailable - google-generativeai not installed")
    except Exception as e:
        raise ConsultantError(f"Gemini fallback failed: {str(e)}")


def ask_consultant(
    message: str,
    intake_data: Optional[Dict[str, Any]] = None,
    programs: Optional[List[Dict[str, Any]]] = None,
    use_fallback: bool = True
) -> str:
    """
    Send a message to the DigitalOcean Agent consultant with user context.
    
    Args:
        message: User's question or message
        intake_data: Optional dictionary with student intake information
        programs: Optional list of program dictionaries with roadmap data
        use_fallback: Whether to fallback to Gemini if DigitalOcean fails (default True)
    
    Returns:
        Consultant's response from DigitalOcean Agent or Gemini fallback
    
    Raises:
        ConsultantError: If API call fails or configuration is missing
    """
    # Get credentials from environment
    agent_endpoint = os.getenv("AGENT_CONSULTANT_ENDPOINT")
    agent_access_key = os.getenv("AGENT_CONSULTANT_ACCESS_KEY")
    
    if not agent_endpoint or not agent_access_key:
        if use_fallback:
            print("DigitalOcean Agent not configured, using Gemini fallback")
            return _try_gemini_fallback(message, intake_data, programs)
        else:
            raise ConsultantError(
                "AGENT_CONSULTANT_ENDPOINT and AGENT_CONSULTANT_ACCESS_KEY must be set in .env file"
            )
    
    # Validate message
    if not message or not message.strip():
        raise ConsultantError("Message cannot be empty")
    
    # Ensure endpoint has correct format
    endpoint = agent_endpoint.rstrip('/')
    if not endpoint.endswith('/api/v1/chat/completions'):
        endpoint = endpoint + '/api/v1/chat/completions'
    
    # Format context from intake data and programs
    context = format_context(intake_data, programs, max_programs=3)
    
    # Build the full prompt
    if context:
        system_message = f"""You are an Ontario university admissions consultant chatbot. You help students plan, clarify requirements, and make progress on their application roadmaps.

{context}

ROLE & GOAL:
Respond to the user's message with practical, accurate advice grounded in the student's data, the current roadmaps, and your knowledge base of Ontario university program requirements.

HARD RULES:
- Be consistent with the roadmaps. Treat roadmap tasks and statuses as the source of truth for what the student has/hasn't done.
- Use your knowledge base. Do not invent admission requirements, deadlines, averages, or policies. If unknown, say what's missing and suggest how to verify.
- Personalize every answer using student context: their grades, missing prerequisites, leadership, co-op preference, and chosen programs.
- Don't overload. Give 3–7 actionable points max, unless the user asks for more.
- When asked "what should I do next?" choose the top 3 next tasks across roadmaps based on urgency (deadlines) and impact (eligibility blockers first).
- If the user asks about a specific program, answer using that program's roadmap + knowledge base requirements.
- Be supportive and precise. Ask at most one follow-up question, only if required to avoid a wrong recommendation.

BEHAVIOR GUIDELINES:
- If the student is missing a prerequisite for a selected program, flag it early and propose a concrete fix (course plan, online accredited option, timeline).
- If the student wants co-op, emphasize co-op programs and tasks like resume/portfolio.
- If leadership level is high (3–4), help them translate it into admissions evidence (impact bullets, references, essay stories).

FORMATTING REQUIREMENTS:
- Write in clear, conversational paragraphs. Do NOT use tables or complex formatting.
- Use simple bullet points (•) or numbered lists when listing action items.
- Keep your response concise and easy to read on mobile devices.
- Structure your response as: brief context paragraph → actionable advice in list format → closing encouragement.
- Example format:
  "Based on your profile, here are the most important steps for [program]:
  
  • First action item with brief explanation
  • Second action item with brief explanation
  • Third action item with brief explanation
  
  These steps will help you [outcome]. Let me know if you need help with any of these!"

Provide personalized, actionable advice based on this student's specific situation."""
        
        # Truncate context if needed to stay within token limits
        context = _truncate_context_if_needed(context, message, system_message)
        
        # Rebuild system message with potentially truncated context
        system_message = f"""You are an Ontario university admissions consultant chatbot. You help students plan, clarify requirements, and make progress on their application roadmaps.

{context}

ROLE & GOAL:
Respond to the user's message with practical, accurate advice grounded in the student's data, the current roadmaps, and your knowledge base of Ontario university program requirements.

HARD RULES:
- Be consistent with the roadmaps. Treat roadmap tasks and statuses as the source of truth for what the student has/hasn't done.
- Use your knowledge base. Do not invent admission requirements, deadlines, averages, or policies. If unknown, say what's missing and suggest how to verify.
- Personalize every answer using student context: their grades, missing prerequisites, leadership, co-op preference, and chosen programs.
- Don't overload. Give 3–7 actionable points max, unless the user asks for more.
- When asked "what should I do next?" choose the top 3 next tasks across roadmaps based on urgency (deadlines) and impact (eligibility blockers first).
- If the user asks about a specific program, answer using that program's roadmap + knowledge base requirements.
- Be supportive and precise. Ask at most one follow-up question, only if required to avoid a wrong recommendation.

BEHAVIOR GUIDELINES:
- If the student is missing a prerequisite for a selected program, flag it early and propose a concrete fix (course plan, online accredited option, timeline).
- If the student wants co-op, emphasize co-op programs and tasks like resume/portfolio.
- If leadership level is high (3–4), help them translate it into admissions evidence (impact bullets, references, essay stories).

FORMATTING REQUIREMENTS:
- Write in clear, conversational paragraphs. Do NOT use tables or complex formatting.
- Use simple bullet points (•) or numbered lists when listing action items.
- Keep your response concise and easy to read on mobile devices.
- Structure your response as: brief context paragraph → actionable advice in list format → closing encouragement.

Provide personalized, actionable advice based on this student's specific situation."""
    else:
        system_message = """You are an Ontario university admissions consultant chatbot for Ontario universities. 
Provide helpful advice about university applications, programs, and admissions.

FORMATTING REQUIREMENTS:
- Write in clear, conversational paragraphs. Do NOT use tables.
- Use simple bullet points (•) or numbered lists when listing action items.
- Keep responses concise and easy to read.
- Structure as: context → actionable advice → encouragement."""
    
    # Build messages array
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": message
        }
    ]
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {agent_access_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "stream": False,
        "include_functions_info": True,
        "include_retrieval_info": True,
        "include_guardrails_info": False
    }
    
    try:
        # Call DigitalOcean Agent API
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=60
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        if not result.get("choices"):
            raise ConsultantError(f"No response from agent. Full response: {result}")
        
        # Extract message content
        message_content = result["choices"][0]["message"]
        answer = message_content.get("content", "")
        
        # If content is empty, check reasoning_content
        if not answer or not answer.strip():
            answer = message_content.get("reasoning_content", "")
        
        if not answer or not answer.strip():
            raise ConsultantError("Agent returned empty response")
        
        return answer
        
    except requests.exceptions.Timeout:
        if use_fallback:
            print("DigitalOcean Agent timed out, using Gemini fallback")
            return _try_gemini_fallback(message, intake_data, programs)
        raise ConsultantError("Request to agent timed out (60s). Please try again.")
    except requests.exceptions.RequestException as e:
        if use_fallback:
            print(f"DigitalOcean Agent failed ({str(e)}), using Gemini fallback")
            return _try_gemini_fallback(message, intake_data, programs)
        raise ConsultantError(f"Failed to connect to agent: {str(e)}")
    except ConsultantError:
        # Don't fallback on ConsultantError (empty response, etc) - these are logic errors
        raise
    except Exception as e:
        if use_fallback:
            print(f"Unexpected error with DigitalOcean Agent ({str(e)}), using Gemini fallback")
            return _try_gemini_fallback(message, intake_data, programs)
        raise ConsultantError(f"Failed to get response from consultant: {str(e)}")


# Backward compatibility: Keep Consultant class for existing code
class Consultant:
    """
    Legacy Consultant class for backward compatibility.
    New code should use ask_consultant() function directly.
    """
    
    def __init__(self):
        """Initialize consultant (no-op for compatibility)"""
        pass
    
    def ask(self, question: str) -> str:
        """
        Ask the consultant a question without context.
        
        Args:
            question: User's question
        
        Returns:
            Consultant's response
        """
        return ask_consultant(question)

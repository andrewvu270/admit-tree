"""
Agent Programs module - Calls DigitalOcean Agent to recommend university programs.
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_ENDPOINT = os.getenv("AGENT_PROGRAMS_ENDPOINT")
AGENT_ACCESS_KEY = os.getenv("AGENT_PROGRAMS_ACCESS_KEY")


class AgentProgramsError(Exception):
    """Custom exception for agent errors"""
    pass


def format_student_profile(profile):
    """
    Format student profile into a prompt that requests matcher-compatible JSON output.
    
    Args:
        profile: Dictionary with keys:
            - grade_level: int
            - average: float
            - wants_coop: bool
            - extra_curriculars: list of tuples (name, level)
            - major_interests: list of strings
            - courses_taken: list of tuples (course_code, grade)
    
    Returns:
        Formatted prompt requesting specific JSON format
    """
    # Format extracurriculars
    ec_text = ""
    if profile.get('extra_curriculars'):
        ec_list = [f"{name} (Level {level})" for name, level in profile['extra_curriculars']]
        ec_text = f"\nExtracurriculars: {', '.join(ec_list)}"
    
    # Format courses
    courses_text = ""
    if profile.get('courses_taken'):
        courses_list = [f"{code} ({grade}%)" for code, grade in profile['courses_taken']]
        courses_text = f"\nCourses Taken: {', '.join(courses_list)}"
    
    # Format interests
    interests_text = ""
    if profile.get('major_interests'):
        interests_text = f"\nInterests: {', '.join(profile['major_interests'])}"
    
    # Build the prompt requesting specific JSON format
    prompt = f"""Student Profile:
Grade Level: {profile.get('grade_level', 'N/A')}
Current Average: {profile.get('average', 'N/A')}%
Wants Co-op: {'Yes' if profile.get('wants_coop') else 'No'}{ec_text}{interests_text}{courses_text}

Based on this student profile and your knowledge base of Ontario university programs, recommend the top 6-10 most suitable programs.

CRITICAL: You MUST return ONLY valid JSON in this EXACT format with NO markdown, NO code fences, NO explanations:

[
  {{
    "university": "University Name",
    "program": "Program Name",
    "score": 85.5,
    "breakdown": {{
      "academic": 0.92,
      "interest": 0.85,
      "ec": 0.78,
      "coop_fit": 1.0
    }}
  }}
]

Requirements:
- Return 6-10 programs ranked by fit (best first)
- score: 0-100 representing overall fit
- breakdown values: 0.0-1.0 for each category
  - academic: how well student meets grade/course requirements
  - interest: alignment with student's interests
  - ec: extracurricular fit
  - coop_fit: 1.0 if co-op matches preference, 0.85-0.95 otherwise
- Use real Ontario universities and programs from your knowledge base
- Consider eligibility (prerequisites, grades) first
- Prefer co-op programs if student wants co-op

OUTPUT ONLY THE JSON ARRAY. NO OTHER TEXT."""
    
    return prompt


def get_program_recommendations(student_profile):
    """
    Call the DigitalOcean Agent to get program recommendations.
    
    Args:
        student_profile: Dictionary with student information
    
    Returns:
        Agent response as a dictionary
    
    Raises:
        AgentProgramsError: If the agent call fails
    """
    if not AGENT_ENDPOINT or not AGENT_ACCESS_KEY:
        raise AgentProgramsError("AGENT_PROGRAMS_ENDPOINT or AGENT_PROGRAMS_ACCESS_KEY not configured in .env")
    
    # Format the student profile into a prompt
    prompt = format_student_profile(student_profile)
    
    # Prepare the request
    headers = {
        "Authorization": f"Bearer {AGENT_ACCESS_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build messages array like the consultant does
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    payload = {
        "messages": messages,
        "stream": False,
        "include_functions_info": True,
        "include_retrieval_info": True,
        "include_guardrails_info": False
    }
    
    try:
        # Ensure endpoint ends with /api/v1/chat/completions
        endpoint = AGENT_ENDPOINT.rstrip('/')
        if not endpoint.endswith('/api/v1/chat/completions'):
            endpoint = endpoint + '/api/v1/chat/completions'
        
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the message content
        if not result.get("choices"):
            raise AgentProgramsError(f"No response from agent. Full response: {result}")
        
        message = result["choices"][0]["message"]
        answer = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        
        # Debug: print raw response
        print(f"Raw agent response (content): {answer if answer else 'EMPTY'}")
        print(f"Raw agent response (reasoning_content): {reasoning[:500] if reasoning else 'EMPTY'}...")
        
        # If content is empty, try to extract JSON from reasoning_content
        if not answer or not answer.strip():
            answer = reasoning
        
        # Also check if reasoning has more complete JSON than content
        if reasoning and len(reasoning) > len(answer):
            # Check if reasoning contains a JSON array
            reasoning_start = reasoning.find("[")
            reasoning_end = reasoning.rfind("]")
            if reasoning_start != -1 and reasoning_end != -1:
                # Use reasoning if it has a JSON array
                answer = reasoning
        
        if not answer or not answer.strip():
            raise AgentProgramsError(f"Agent returned empty response. Try submitting again.")
        
        # Parse the JSON response from the agent
        try:
            # Remove markdown code fences if present
            if "```json" in answer:
                start = answer.find("```json") + 7
                end = answer.find("```", start)
                answer = answer[start:end].strip()
            elif "```" in answer:
                start = answer.find("```") + 3
                end = answer.find("```", start)
                answer = answer[start:end].strip()
            
            # Find JSON array in the response (look for [ ... ])
            start_idx = answer.find("[")
            end_idx = answer.rfind("]")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = answer[start_idx:end_idx+1]
                
                # Try to parse it
                rankings = json.loads(json_str)
                
                if not isinstance(rankings, list):
                    raise AgentProgramsError(f"Expected JSON array, got {type(rankings)}")
                
                if len(rankings) == 0:
                    raise AgentProgramsError("Agent returned empty rankings array")
                
                return rankings
            else:
                # No JSON array found - agent is still reasoning
                # Check if there's a partial array being built
                if "university" in answer.lower() and "program" in answer.lower():
                    raise AgentProgramsError(f"Agent is still generating response. Please try again in a moment.")
                else:
                    raise AgentProgramsError(f"No JSON array found in response. Agent may still be reasoning. Try again.")
            
        except json.JSONDecodeError as e:
            raise AgentProgramsError(f"Failed to parse agent JSON response: {str(e)}. Try submitting again.")
        
    except requests.exceptions.Timeout:
        raise AgentProgramsError("Agent request timed out")
    except requests.exceptions.RequestException as e:
        raise AgentProgramsError(f"Agent request failed: {str(e)}")
    except ValueError as e:
        raise AgentProgramsError(f"Failed to parse agent response: {str(e)}")

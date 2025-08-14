import google.generativeai as genai
import json
from config import API_KEY

# Initialize AI Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_roadmap(role):
    """Generate career roadmap as JSON"""
    prompt = f"""
    Generate a roadmap for becoming a {role}. 
    Format it as JSON with 'title', 'description', and 'steps' containing 'title' and 'description'.

    Example:
    {{
        "title": "Roadmap to Becoming a {role}",
        "description": "A structured path to become a {role}.",
        "steps": [
            {{"title": "Learn Fundamentals", "description": "Master basics of programming."}},
            {{"title": "SQL Mastery", "description": "Work with databases efficiently."}},
            {{"title": "Python for Data Analysis", "description": "Use Python to analyze data."}},
            {{"title": "Tableau/Power BI", "description": "Create interactive dashboards."}},
            {{"title": "Real-world Project", "description": "Build a portfolio project."}}
        ]
    }}
    """

    response = model.generate_content(prompt)

    try:
        return json.loads(response.text.strip())  # Ensure valid JSON
    except json.JSONDecodeError:
        return {"error": "Invalid AI response."}

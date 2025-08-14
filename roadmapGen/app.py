from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pandas as pd
from rapidfuzz import process
import os
from config import API_KEY
import time
import random
import re

app = Flask(__name__)

# Configure API key
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    print(f"❌ Error configuring Gemini AI: {e}")
    raise

# Load Course Dataset with Error Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_DATA_PATH = os.path.join(BASE_DIR, "data", "courses.csv")

try:
    if not os.path.exists(COURSE_DATA_PATH):
        raise FileNotFoundError(f"Course data file not found at {COURSE_DATA_PATH}")
    courses_df = pd.read_csv(COURSE_DATA_PATH, encoding="utf-8", on_bad_lines="skip")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    courses_df = pd.DataFrame(columns=["Course Name", "Description", "Link", "Rating", "Enrollments"])  # Empty fallback

def generate_roadmap(role):
    """Generate a structured learning roadmap using Gemini AI."""
    prompt = f"""
    Generate a structured learning roadmap to become a {role}. 
    The roadmap should have major milestones separated by ' ➝ '. 
    Example: 'Starting Point ➝ Statistics ➝ Python ➝ SQL ➝ Excel ➝ Data Visualization ➝ BI Tools ➝ Being Awesome!'
    """
    
    # Initialize retry parameters
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Initial backoff in seconds
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            return response.text.strip() if hasattr(response, 'text') else "❌ Invalid response from AI."
        except Exception as e:
            if "ResourceExhausted" in str(e) or "429" in str(e):
                retry_count += 1
                if retry_count >= max_retries:
                    return "Sorry, we're experiencing high demand. Please try again in a minute."
                
                # Add jitter to backoff time to prevent all clients retrying simultaneously
                sleep_time = backoff_time + (random.random() * 0.5)
                print(f"API quota exceeded. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                backoff_time *= 2  # Exponential backoff
            else:
                # For other exceptions, don't retry
                return f"Error generating roadmap: {str(e)}"
    
    return "Unable to generate roadmap due to service limitations. Please try again later."

def get_top_courses(step, top_n=5, threshold=60):
    """Get the top N trending courses based on fuzzy matching."""
    if courses_df.empty:
        return []

    # Find top N best matches using RapidFuzz
    matches = process.extract(step, courses_df["Course Name"].dropna().tolist(), limit=top_n)

    # Filter based on match quality
    matched_courses = [match[0] for match in matches if match[1] >= threshold]

    if matched_courses:
        filtered_df = courses_df[courses_df["Course Name"].isin(matched_courses)]

        # Sort courses by rating and enrollments if available
        if "Rating" in filtered_df.columns and "Enrollments" in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=["Rating", "Enrollments"], ascending=[False, False])

        return filtered_df.head(top_n).to_dict(orient="records")

    return []

@app.route("/", methods=["GET", "POST"])
def index():
    roadmap = None
    if request.method == "POST":
        role = request.form.get("role")
        roadmap = generate_roadmap(role)

    return render_template("index.html", roadmap=roadmap)

@app.route("/roadmap/<step>")
def roadmap_step(step):
    """Render a detailed explanation page for a roadmap step with top recommended courses."""
    try:
        recommended_courses = get_top_courses(step, top_n=5, threshold=50)  # Lower threshold to find more matches
        roadmap = request.args.get("roadmap", "")
        if isinstance(roadmap, str):
            steps = roadmap.split(" ➝ ")
        else:
            steps = []
        return render_template("roadmap_step.html", step=step, recommended_courses=recommended_courses, steps=steps, roadmap=roadmap)
    except Exception as e:
        print(f"Error retrieving courses: {e}")
        # Return with empty recommended courses
        roadmap = request.args.get("roadmap", "")
        if isinstance(roadmap, str):
            steps = roadmap.split(" ➝ ")
        else:
            steps = []
        return render_template("roadmap_step.html", step=step, recommended_courses=[], steps=steps, roadmap=roadmap, error=str(e))

@app.route("/generate-test", methods=["POST"])
def generate_test():
    """Generate test questions for a specific topic."""
    data = request.get_json()
    topic = data.get('topic', '')
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    prompt = f"""
    Generate a test to assess knowledge level for {topic}. 
    
    Create exactly 20 multiple-choice questions (MCQs) related to {topic} concepts. 
    
    Each question should:
    1. Cover an important concept in {topic}
    2. Have exactly 4 answer options (A, B, C, D)
    3. Clearly indicate which option is correct
    
    Format each question as follows:
    1. [Question text]
       A) [Option A]
       B) [Option B]
       C) [Option C]
       D) [Option D]
       Correct Answer: [Letter] (must be one of A, B, C, or D)
    
    Make sure to give just the questions without any other text.
    """
    
    # Initialize retry parameters
    max_retries = 3
    retry_count = 0
    backoff_time = 2
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            
            if not response or not hasattr(response, 'text'):
                return jsonify({"error": "Invalid response from AI"}), 500
            
            # Parse the response to extract questions
            questions = parse_questions_from_text(response.text)
            
            if not questions:
                return jsonify({"error": "Failed to parse questions from response"}), 500
            
            return jsonify({"questions": questions})
            
        except Exception as e:
            if "ResourceExhausted" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                retry_count += 1
                if retry_count >= max_retries:
                    return jsonify({"error": "Service temporarily unavailable due to high demand. Please try again later."}), 429
                
                sleep_time = backoff_time + (random.random() * 0.5)
                print(f"API quota exceeded. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                backoff_time *= 2
            else:
                return jsonify({"error": f"Error generating test: {str(e)}"}), 500
    
    return jsonify({"error": "Unable to generate test due to service limitations. Please try again later."}), 503

def parse_questions_from_text(text):
    """Parse questions from the AI response text."""
    questions = []
    lines = text.strip().split('\n')
    current_question = None
    options = []
    correct_answer = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if it's a question (starts with a number)
        if re.match(r'^\d+\.', line):
            # Save previous question if it exists
            if current_question and len(options) == 4 and correct_answer:
                questions.append({
                    "question": current_question,
                    "options": options,
                    "correctAnswer": correct_answer
                })
            
            # Start new question
            current_question = re.sub(r'^\d+\.\s*', '', line)
            options = []
            correct_answer = None
            
        # Check if it's an option
        elif re.match(r'^[A-D]\)', line):
            option_text = re.sub(r'^[A-D]\)\s*', '', line)
            options.append(option_text)
            
        # Check if it's the correct answer
        elif line.startswith('Correct Answer:'):
            answer_letter = re.search(r'Correct Answer:\s*([A-D])', line)
            if answer_letter:
                correct_answer = answer_letter.group(1)
    
    # Don't forget the last question
    if current_question and len(options) == 4 and correct_answer:
        questions.append({
            "question": current_question,
            "options": options,
            "correctAnswer": correct_answer
        })
    
    return questions

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Use default port

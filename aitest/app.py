import fitz  # PyMuPDF for text extraction
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import re
import time
import random

app = Flask(__name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyARqUGRLuL9U2DjBs1C8JkUUAUtGB3fGPU")


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume."""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        
        # Clean up the text
        text = text.strip()
        if not text:
            return "No text content found in the uploaded PDF file."
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error reading PDF file: {str(e)}"


def generate_mcqs(resume_text):
    """Generates 20 MCQs testing knowledge based on resume analysis."""
    prompt = f"""Generate EXACTLY 20 multiple choice questions based on the resume content provided.

    **IMPORTANT: Use this EXACT format for every question:**

    1. What is the primary benefit of using React hooks over class components?
    A) Better performance optimization
    B) Easier state management and lifecycle handling
    C) Automatic error boundaries
    D) Built-in testing capabilities
    Correct Answer: B

    2. In Node.js development, which approach is best for handling database connections?
    A) Create new connection for each request
    B) Use connection pooling with proper error handling
    C) Keep a single global connection
    D) Use synchronous database operations
    Correct Answer: B

    **REQUIREMENTS:**
    1. Base ALL questions on specific technologies/skills/projects mentioned in the resume
    2. Use medium difficulty - focus on practical implementation, not basic definitions
    3. Each question must have exactly 4 options labeled A) B) C) D)
    4. Each question must end with "Correct Answer: [Letter]"
    5. Number each question from 1 to 20
    6. Questions should cover: programming languages, frameworks, databases, tools, project technologies

    **Resume Content:**
    {resume_text}

    Generate exactly 20 questions now using the format shown above:"""

    # Add retry logic for API quota issues
    max_retries = 3
    retry_count = 0
    backoff_time = 2
    
    while retry_count < max_retries:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            
            mcqs = []
            if response and response.text:
                text = response.text.strip()
                print(f"AI Response length: {len(text)} characters")
                
                # First, let's save the raw response for debugging
                import re
                
                # Method 1: Parse using regex to find complete question blocks
                question_pattern = r'(\d+\..*?)(?=\d+\.|$)'
                question_matches = re.findall(question_pattern, text, re.DOTALL)
                
                print(f"Method 1: Found {len(question_matches)} question blocks")
                
                for i, match in enumerate(question_matches):
                    if len(mcqs) >= 20:  # Stop at 20 questions
                        break
                        
                    # Split into lines and clean
                    lines = [line.strip() for line in match.split('\n') if line.strip()]
                    
                    if len(lines) < 5:  # Need at least question + 4 options (correct answer might be missing)
                        continue
                    
                    # Extract question (remove numbering)
                    question_line = lines[0]
                    question = re.sub(r'^\d+\.\s*', '', question_line).strip()
                    
                    if not question or len(question) < 10:  # Skip if question is too short
                        continue
                    
                    # Find options and correct answer
                    options = []
                    correct_answer_line = ""
                    
                    for line in lines[1:]:
                        if re.match(r'^[A-D]\)\s*', line):
                            option_text = re.sub(r'^[A-D]\)\s*', '', line).strip()
                            if option_text:  # Only add non-empty options
                                options.append(option_text)
                        elif 'correct answer' in line.lower():
                            correct_answer_line = line
                    
                    # Only proceed if we have exactly 4 options
                    if len(options) == 4:
                        # Extract correct answer letter
                        correct_letter = 'A'  # Default fallback
                        
                        if correct_answer_line:
                            # Look for answer pattern
                            answer_match = re.search(r'correct answer:\s*([A-D])', correct_answer_line, re.IGNORECASE)
                            if answer_match:
                                correct_letter = answer_match.group(1).upper()
                            else:
                                # Look for just the letter
                                letter_match = re.search(r'([A-D])', correct_answer_line)
                                if letter_match:
                                    correct_letter = letter_match.group(1).upper()
                        
                        # Validate correct answer
                        correct_index = ord(correct_letter) - ord('A')
                        if 0 <= correct_index < 4:
                            correct_answer = f"{correct_letter}) {options[correct_index]}"
                            
                            mcq = {
                                "question": question,
                                "options": [f"{chr(65+j)}) {opt}" for j, opt in enumerate(options)],
                                "correct_answer": correct_answer
                            }
                            mcqs.append(mcq)
                            print(f"Parsed question {len(mcqs)}: {question[:50]}...")
                
                # Method 2: If we don't have enough questions, try line-by-line parsing
                if len(mcqs) < 15:
                    print(f"Method 1 got {len(mcqs)} questions. Trying method 2...")
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    
                    i = 0
                    while i < len(lines) and len(mcqs) < 20:
                        line = lines[i]
                        
                        # Look for question patterns
                        if (re.match(r'^\d+\.', line) or ('?' in line and not line.startswith(('A)', 'B)', 'C)', 'D)')))):
                            question = re.sub(r'^\d+\.\s*', '', line).strip()
                            
                            if len(question) < 10:  # Skip short questions
                                i += 1
                                continue
                            
                            # Look for options in the next lines
                            options = []
                            j = i + 1
                            
                            while j < len(lines) and len(options) < 4:
                                if re.match(r'^[A-D]\)', lines[j]):
                                    option_text = re.sub(r'^[A-D]\)\s*', '', lines[j]).strip()
                                    if option_text:
                                        options.append(option_text)
                                elif len(options) > 0:  # Stop if we hit non-option after starting options
                                    break
                                j += 1
                            
                            # If we have 4 options, create the question
                            if len(options) == 4:
                                # Look for correct answer in next few lines
                                correct_letter = 'A'  # Default
                                k = j
                                while k < len(lines) and k < j + 3:
                                    if 'correct answer' in lines[k].lower():
                                        answer_match = re.search(r'([A-D])', lines[k])
                                        if answer_match:
                                            correct_letter = answer_match.group(1).upper()
                                        break
                                    k += 1
                                
                                # Add the question
                                correct_index = ord(correct_letter) - ord('A')
                                if 0 <= correct_index < 4:
                                    correct_answer = f"{correct_letter}) {options[correct_index]}"
                                    
                                    # Check if we already have this question (avoid duplicates)
                                    is_duplicate = False
                                    for existing_mcq in mcqs:
                                        if existing_mcq['question'][:30] == question[:30]:
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        mcq = {
                                            "question": question,
                                            "options": [f"{chr(65+idx)}) {opt}" for idx, opt in enumerate(options)],
                                            "correct_answer": correct_answer
                                        }
                                        mcqs.append(mcq)
                                        print(f"Method 2 parsed question {len(mcqs)}: {question[:50]}...")
                                
                                i = k  # Skip to after this question block
                            else:
                                i += 1
                        else:
                            i += 1
                
                print(f"Total parsed questions: {len(mcqs)}")
                
                # Method 3: If still not enough, create some basic questions
                if len(mcqs) < 10:
                    print("Generating fallback questions...")
                    fallback_questions = [
                        {
                            "question": "Which programming concept is most important for software development?",
                            "options": ["A) Object-oriented programming", "B) Functional programming", "C) Procedural programming", "D) All of the above"],
                            "correct_answer": "D) All of the above"
                        },
                        {
                            "question": "What is the best practice for code documentation?",
                            "options": ["A) Write comments for every line", "B) Document complex logic and APIs", "C) Never write comments", "D) Only document at the end"],
                            "correct_answer": "B) Document complex logic and APIs"
                        },
                        {
                            "question": "Which version control practice is most recommended?",
                            "options": ["A) Commit all changes at once", "B) Make frequent, small commits", "C) Never use version control", "D) Only commit when project is complete"],
                            "correct_answer": "B) Make frequent, small commits"
                        }
                    ]
                    
                    for fallback in fallback_questions:
                        if len(mcqs) < 20:
                            mcqs.append(fallback)
            
            # Return questions if we have any, otherwise return fallback
            if len(mcqs) > 0:
                return mcqs[:20]  # Return maximum 20 questions
            else:
                print("No questions could be parsed, returning fallback question")
                return [{
                    "question": "Based on your resume, which technology would be most suitable for your next project?",
                    "options": ["A) Frontend technologies", "B) Backend technologies", "C) Full-stack development", "D) Mobile development"],
                    "correct_answer": "C) Full-stack development"
                }]
        except Exception as e:
            if "ResourceExhausted" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                retry_count += 1
                if retry_count >= max_retries:
                    return [{"question": "Service temporarily unavailable. Please try again later.", 
                            "options": ["A", "B", "C", "D"], "correct_answer": "A"}]
                
                sleep_time = backoff_time + (random.random() * 0.5)
                print(f"API quota exceeded. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                backoff_time *= 2
            else:
                return [{"question": f"Error generating questions: {str(e)}", 
                        "options": ["A", "B", "C", "D"], "correct_answer": "A"}]
    
    return [{"question": "Unable to generate questions due to service limitations.", 
            "options": ["A", "B", "C", "D"], "correct_answer": "A"}]


def assign_nft_tag(score):
    """Assigns NFT tag based on the candidate's score."""
    if score >= 90:
        return "Platinum NFT"
    elif score >= 70:
        return "Gold NFT"
    elif score >= 50:
        return "Silver NFT"
    else:
        return "No NFT"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handles file upload and generates MCQs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please upload a PDF file only"}), 400

    # Ensure uploads directory exists
    import os
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    
    # Extract text from PDF
    resume_text = extract_text_from_pdf(file_path)
    
    # Check if text extraction was successful
    if "Error reading PDF file" in resume_text or "No text content found" in resume_text:
        return jsonify({"error": resume_text}), 400
    
    if len(resume_text.strip()) < 100:  # Minimum reasonable resume length
        return jsonify({"error": "Resume content seems too short. Please ensure your PDF contains readable text."}), 400

    # Generate MCQs based on resume analysis
    print(f"Generating MCQs for resume with {len(resume_text)} characters of text...")
    mcqs = generate_mcqs(resume_text)
    
    if not mcqs or len(mcqs) == 0:
        return jsonify({"error": "Failed to generate questions from resume. Please try again."}), 500
    
    print(f"Generated {len(mcqs)} MCQ questions")
    return jsonify({"mcqs": mcqs, "message": f"Successfully generated {len(mcqs)} questions based on your resume analysis."})


@app.route("/evaluate", methods=["POST"])
def evaluate_answers():
    """Evaluates candidate MCQ answers and assigns an NFT tag."""
    data = request.json
    mcqs = data.get("mcqs", [])
    answers = data.get("answers", [])

    if not mcqs or not answers or len(mcqs) != len(answers):
        return jsonify({"error": "Invalid input"}), 400

    correct_count = sum(1 for i in range(len(mcqs)) if mcqs[i]["correct_answer"].startswith(answers[i]))
    score = (correct_count / len(mcqs)) * 100
    nft_tag = assign_nft_tag(score)

    return jsonify({"score": f"{score:.2f}%", "nft_tag": nft_tag})


if __name__ == "__main__":
    app.run(debug=True,port=9000)

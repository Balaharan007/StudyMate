from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
import io
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "hackrx-document-intelligence-secret")

# Configuration
API_BASE_URL = f"http://localhost:{os.getenv('API_PORT', 8000)}"
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_api_connection():
    """Test if the API server is running and accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(endpoint: str, data: dict = None, method: str = "GET", files=None):
    """Make API request with authentication"""
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }
    
    if not files:
        headers["Content-Type"] = "application/json"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, headers={"Authorization": f"Bearer {BEARER_TOKEN}"})
            else:
                response = requests.post(url, json=data, headers=headers)
        else:
            response = requests.get(url, headers=headers)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

@app.route('/')
def index():
    """Main dashboard"""
    api_status = test_api_connection()
    return render_template('index.html', api_status=api_status, api_url=API_BASE_URL)

@app.route('/document-upload')
def document_upload():
    """Document upload page with workflow steps"""
    return render_template('document_upload.html')

@app.route('/query-interface')
def query_interface():
    """Query interface page"""
    return render_template('query_interface.html')

@app.route('/system-stats')
def system_stats():
    """System statistics page"""
    stats = make_api_request("/stats")
    documents = make_api_request("/documents")
    return render_template('system_stats.html', stats=stats, documents=documents)

@app.route('/api/document/upload', methods=['POST'])
def upload_document():
    """STEP 1: Upload Document - Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "File type not allowed"}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # STEP 2: Semantic Chunking & Embedding
        result = make_api_request("/document/process", {
            "file_path": file_path,
            "filename": filename,
            "model": GEMINI_MODEL
        }, "POST")
        
        if result:
            return jsonify({
                "success": True,
                "message": "Document processed successfully",
                "document_id": result.get("document_id"),
                "chunks": result.get("chunks_count", 0),
                "embeddings": result.get("embeddings_count", 0)
            })
        else:
            return jsonify({"success": False, "message": "Document processing failed"}), 500
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Upload failed: {str(e)}"}), 500

@app.route('/api/query/process', methods=['POST'])
def process_query():
    """STEP 3-5: Query Processing with workflow steps"""
    data = request.json
    document_id = data.get('document_id')
    query = data.get('query')
    
    if not document_id or not query:
        return jsonify({"success": False, "message": "Document ID and query are required"}), 400
    
    try:
        # STEP 3: Query Parsing (Structured Field Extraction)
        parsed_query = make_api_request("/query/parse", {
            "query": query,
            "model": GEMINI_MODEL
        }, "POST")
        
        if not parsed_query:
            return jsonify({"success": False, "message": "Query parsing failed"}), 500
        
        # STEP 4: Semantic Clause Matching
        semantic_matches = make_api_request("/document/search", {
            "document_id": document_id,
            "query": query,
            "parsed_fields": parsed_query.get("fields", [])
        }, "POST")
        
        if not semantic_matches:
            return jsonify({"success": False, "message": "Semantic matching failed"}), 500
        
        # STEP 5: Decision Logic & LLM Reasoning
        final_result = make_api_request("/query/reason", {
            "query": query,
            "parsed_query": parsed_query,
            "semantic_matches": semantic_matches,
            "model": GEMINI_MODEL
        }, "POST")
        
        if final_result:
            return jsonify({
                "success": True,
                "query": query,
                "parsed_query": parsed_query,
                "semantic_matches": semantic_matches,
                "answer": final_result.get("answer"),
                "confidence": final_result.get("confidence"),
                "sources": final_result.get("sources", [])
            })
        else:
            return jsonify({"success": False, "message": "Query processing failed"}), 500
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Processing failed: {str(e)}"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    backend_status = test_api_connection()
    return jsonify({
        "status": "healthy" if backend_status else "unhealthy",
        "backend_api": "online" if backend_status else "offline",
        "version": "3.0.0"
    })

@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Upload and process document files"""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    file = request.files['file']
    title = request.form.get('title', '')
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Read file content
        file_content = file.read()
        
        # Prepare files for API
        files = {'file': (filename, file_content, file.content_type)}
        data = {'title': title}
        
        result = make_api_request("/documents/upload-file", data, "POST", files)
        
        if result and result.get("success"):
            return jsonify({
                "success": True,
                "message": "File uploaded and processed successfully!",
                "document_id": result.get("document_id"),
                "chunks": result.get("chunks"),
                "filename": filename
            })
        else:
            error_msg = "File processing failed"
            if result and "error" in result:
                error_msg = result["error"]
            elif not result:
                error_msg = "Cannot connect to backend API. Please ensure the backend server is running."
            
            return jsonify({
                "success": False,
                "message": error_msg
            }), 500
    else:
        return jsonify({
            "success": False,
            "message": "Invalid file type. Please upload PDF, DOCX, or TXT files."
        }), 400

@app.route('/api/upload-url', methods=['POST'])
def upload_url():
    """Upload document from URL"""
    data = request.json
    url = data.get('url')
    title = data.get('title', '')
    
    if not url:
        return jsonify({"success": False, "message": "URL is required"}), 400
    
    request_data = {
        "url": url,
        "title": title
    }
    
    result = make_api_request("/documents/upload-url", request_data, "POST")
    
    if result and result.get("success"):
        return jsonify({
            "success": True,
            "message": "Document processed successfully from URL!",
            "document_id": result.get("document_id"),
            "chunks": result.get("chunks")
        })
    else:
        return jsonify({
            "success": False,
            "message": result.get("error", "URL processing failed")
        }), 500

@app.route('/api/query', methods=['POST'])
def query_document():
    """Query documents"""
    data = request.json
    question = data.get('question')
    document_id = data.get('document_id')
    document_url = data.get('document_url')
    
    if not question:
        return jsonify({"success": False, "message": "Question is required"}), 400
    
    request_data = {
        "question": question,
        "document_id": document_id,
        "document_url": document_url
    }
    
    result = make_api_request("/query", request_data, "POST")
    
    if result:
        return jsonify({
            "success": True,
            "answer": result.get("answer"),
            "reasoning": result.get("reasoning", ""),
            "relevant_chunks": result.get("relevant_chunks", []),
            "document_title": result.get("document_title")
        })
    else:
        return jsonify({
            "success": False,
            "message": "Query processing failed"
        }), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of processed documents"""
    documents = make_api_request("/documents")
    return jsonify(documents if documents else [])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = make_api_request("/stats")
    return jsonify(stats if stats else {})

if __name__ == '__main__':
    print("🚀 Starting Document Intelligence Web Application...")
    print("🌐 Flask Web UI will be available at: http://localhost:5555 ")
    print("🔗 Make sure the FastAPI backend is running at:", API_BASE_URL)
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('FLASK_PORT', 5555)),
        debug=True
    )

# 🚀 HackRx 6.0 Document Intelligence Agent - Running Guide

## 📋 Quick Start (Both servers are already running!)

### ✅ Status Check

- **Backend API**: ✅ Running on http://localhost:8000
- **Web Interface**: ✅ New Flask UI on http://localhost:5000
- **Streamlit UI**: ✅ Legacy interface on http://localhost:8501

## 🔧 How to Run the Application

### Method 1: Complete Setup (Backend + New Web UI)

1. **Start Backend API Server**:

```powershell
cd "C:\Users\haran\OneDrive\Desktop\CODE_ALPHA\CAREER_NAVIGATOR_AI\Asking_question\"
python main_final.py
```

2. **Start New Flask Web UI** (in new terminal):

```powershell
cd "C:\Users\haran\OneDrive\Desktop\CODE_ALPHA\CAREER_NAVIGATOR_AI\Asking_question"


python flask_app.py


### Method 2: Legacy Streamlit UI

```powershell
cd "c:\Users\haran\OneDrive\Desktop\new_feature"
streamlit run streamlit_app_v2.py --server.port 8501
```

### Method 3: API Only (for HackRx submission)

```powershell
cd "c:\Users\haran\OneDrive\Desktop\new_feature"
python main_final.py
```

### Method 4: Using Uvicorn (Production)

```powershell
cd "c:\Users\haran\OneDrive\Desktop\new_feature"
uvicorn main_final:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 Access Points

### 1. **New Flask Web UI** (🌟 Recommended - Advanced Styling)

- **URL**: http://localhost:5000
- **Features**:
  - 🎨 Advanced HTML/CSS styling with animations
  - 📱 Responsive Bootstrap design
  - 🚀 Modern UI/UX with gradients and effects
  - 📄 File upload (PDF, DOCX, TXT)
  - ❓ Interactive Q&A interface
  - 📊 System statistics dashboard
  - 🔍 HackRx API testing interface

### 2. **Streamlit Web UI** (Legacy Interface)

- **URL**: http://localhost:8501
- **Features**:
  - 📄 File upload (PDF, DOCX, TXT)
  - ❓ Interactive Q&A
  - 📊 Document statistics
  - 🔍 HackRx API testing

### 3. **FastAPI Backend**

- **URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🧪 Testing Your Document (Resume)

### Option A: Using New Flask Web UI

1. Open http://localhost:5000
2. Go to "Upload Documents" page
3. Upload your `MAHENDRAVIKAS_RESUME.pdf`
4. Navigate to "Query Interface"
5. Select your uploaded document
6. Ask questions like:
   - "What are the key skills mentioned?"
   - "What is the work experience?"
   - "What education background is mentioned?"

### Option B: Using Legacy Streamlit UI

1. Open http://localhost:8501
2. Go to "📄 Document Upload" page
3. Upload your `MAHENDRAVIKAS_RESUME.pdf`
4. Ask questions like:
   - "What are the key skills mentioned?"
   - "What is the work experience?"
   - "What education background is mentioned?"

### Option C: Using API Directly

```powershell
# Test with your resume
python -c "
import requests

# Upload file
files = {'file': open('MAHENDRAVIKAS_RESUME.pdf', 'rb')}
headers = {'Authorization': 'Bearer YOUR_BEARER_TOKEN'}

upload_resp = requests.post('http://localhost:8000/documents/upload-file', headers=headers, files=files)
print('Upload:', upload_resp.json())

# Query the document
query_data = {
    'question': 'What are the key skills of this person?',
    'document_id': upload_resp.json()['document_id']
}
headers['Content-Type'] = 'application/json'
query_resp = requests.post('http://localhost:8000/query', headers=headers, json=query_data)
print('Answer:', query_resp.json()['answer'])
"
```

## 🎯 HackRx API Testing

### Test with Official HackRx Endpoint

```powershell
python -c "
import requests
import json

headers = {
    'Authorization': 'Bearer YOUR_BEARER_TOKEN',
    'Content-Type': 'application/json'
}

# HackRx format
data = {
    'documents': 'This is a sample insurance policy. Coverage includes medical expenses up to $100,000. Premium is $500 annually.',
    'questions': [
        'What type of insurance is this?',
        'What is the coverage amount?',
        'What is the annual premium?'
    ]
}

response = requests.post('http://localhost:8000/hackrx/run', headers=headers, json=data)
print('HackRx Response:', response.json())
"
```

## 📊 Available Endpoints

| Endpoint                 | Method | Purpose                  |
| ------------------------ | ------ | ------------------------ |
| `/`                      | GET    | Service information      |
| `/health`                | GET    | Health check             |
| `/stats`                 | GET    | System statistics        |
| `/hackrx/run`            | POST   | **Main HackRx endpoint** |
| `/documents/upload-file` | POST   | Upload documents         |
| `/query`                 | POST   | Query documents          |
| `/docs`                  | GET    | Swagger documentation    |

## 🔐 Authentication

All endpoints require Bearer token:

```
Authorization: Bearer YOUR_BEARER_TOKEN
```

## 🛠 Troubleshooting

### If Backend Won't Start:

```powershell
# Check if port is in use
netstat -ano | findstr :8000

# Install dependencies
pip install -r requirements.txt

# Check environment variables
python -c "import os; print('GEMINI_API_KEY:', os.getenv('GEMINI_API_KEY')[:10]+'...' if os.getenv('GEMINI_API_KEY') else 'Missing')"
```

### If Streamlit Won't Start:

```powershell
# Try different port
streamlit run streamlit_app_v2.py --server.port 8502

# Check Streamlit installation
streamlit --version
```

## 🎉 Ready for HackRx 6.0!

Your application is now ready for the HackRx 6.0 competition with:

- ✅ Document processing (PDF, DOCX, TXT)
- ✅ AI-powered Q&A with Gemini
- ✅ HackRx API compliance
- ✅ File upload support
- ✅ Semantic search
- ✅ Explainable answers
- ✅ Multi-domain support (Insurance, Legal, HR, Compliance)

## 📞 Quick Commands Summary

```powershell
# Start everything (recommended)
cd "c:\Users\haran\OneDrive\Desktop\new_feature"

# Terminal 1: Backend API
python main_final.py

# Terminal 2: New Flask Web UI
start_web_app.bat
# OR manually: python flask_app.py

# Terminal 3 (Optional): Legacy Streamlit UI
streamlit run streamlit_app_v2.py

# Then open:
# - New Web UI: http://localhost:5000 (🌟 Recommended)
# - Legacy UI: http://localhost:8501
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

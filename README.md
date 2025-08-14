# HackRx 6.0 Document Intelligence Agent

A comprehensive AI-powered document analysis and query system built for HackRx 6.0 hackathon.

## 🌟 Features

- **Document Processing**: Supports PDF and DOCX files
- **Semantic Search**: Vector-based similarity search using Pinecone
- **AI-Powered Answers**: Uses Google Gemini for intelligent responses
- **RESTful API**: FastAPI backend with authentication
- **Interactive UI**: Streamlit-based user interface
- **Database Storage**: PostgreSQL for document and query management
- **HackRx Compatible**: Implements the required `/hackrx/run` endpoint

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │   PostgreSQL    │
│                 │◄──►│                 │◄──►│                 │
│  User Interface │    │   Document      │    │   Metadata      │
└─────────────────┘    │   Processing    │    │   Storage       │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Pinecone DB   │    │  Google Gemini  │
                       │                 │    │                 │
                       │  Vector Storage │    │  LLM Reasoning  │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database
- API keys for:
  - Pinecone
  - Google Gemini
  - HuggingFace (optional)

### Installation

1. **Clone or download the project**
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** in `.env`:

   ```env
   # Your API keys are already configured in the .env file
   ```

4. **Start the application**:

   **Windows:**

   ```cmd
   start.bat
   ```

   **Linux/Mac:**

   ```bash
   chmod +x start.sh
   ./start.sh
   ```

5. **Access the applications**:
   - Streamlit UI: http://localhost:8501
   - FastAPI docs: http://localhost:8000/docs
   - HackRx endpoint: http://localhost:8000/hackrx/run

## 📡 API Endpoints

### Main HackRx Endpoint

```http
POST /hackrx/run
Authorization: Bearer YOUR_BEARER_TOKEN
Content-Type: application/json

{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response:**

```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "Yes, the policy covers maternity expenses..."
  ]
}
```

### Other Endpoints

- `POST /documents/upload` - Upload and process documents
- `POST /query` - Query specific documents
- `GET /documents` - List all documents
- `GET /documents/{id}/queries` - Get document queries

## 🧠 How It Works

1. **Document Ingestion**:

   - Downloads documents from URLs
   - Extracts text using PyPDF2/python-docx
   - Chunks text into meaningful segments
   - Generates embeddings using SentenceTransformers

2. **Vector Storage**:

   - Stores embeddings in Pinecone vector database
   - Enables fast semantic similarity search

3. **Query Processing**:

   - Converts questions to embeddings
   - Finds relevant document chunks
   - Uses Google Gemini for answer generation

4. **Response Generation**:
   - Provides accurate, contextual answers
   - References specific document clauses
   - Returns structured JSON responses

## 📁 Project Structure

```
hackrx-document-agent/
├── main.py                 # FastAPI application
├── streamlit_app.py        # Streamlit user interface
├── document_processor.py   # Core document processing logic
├── database.py            # PostgreSQL database models
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── start.bat             # Windows startup script
├── start.sh              # Linux/Mac startup script
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

All required environment variables are pre-configured in the `.env` file:

- **Pinecone**: Vector database for embeddings
- **Gemini**: LLM for answer generation
- **PostgreSQL**: Metadata storage
- **Bearer Token**: API authentication

### Customization

- **Chunk Size**: Modify `chunk_size` in `document_processor.py`
- **Embedding Model**: Change model in `DocumentProcessor.__init__()`
- **Vector Search**: Adjust `top_k` parameter for relevance
- **API Settings**: Update ports and hosts in `.env`

## 🚀 Deployment Options

### Recommended Free Platforms:

1. **Render** (Recommended) ⭐

   - Easy deployment
   - Free PostgreSQL
   - HTTPS included
   - Good for FastAPI

2. **Railway**

   - Simple setup
   - Database included
   - Auto-deployment

3. **Heroku**

   - Well-documented
   - Add-ons available
   - Reliable

4. **Vercel** (Functions)
   - Serverless deployment
   - Good for APIs
   - Fast CDN

### Deployment Steps (Render):

1. Push code to GitHub
2. Connect Render to your repository
3. Add environment variables
4. Deploy both FastAPI and Streamlit services
5. Configure PostgreSQL add-on

## 🧪 Testing

### Test with Sample Data

The HackRx sample document and questions are pre-configured in the Streamlit interface:

1. Open http://localhost:8501
2. Go to "HackRx API Test"
3. Click "Run HackRx API"
4. View results

### Manual API Testing

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## 🔍 Troubleshooting

### Common Issues:

1. **Pinecone Connection**: Check API key and region
2. **PostgreSQL**: Ensure database is running and accessible
3. **Gemini API**: Verify API key and quota
4. **Document Download**: Check URL accessibility and format

### Debug Mode:

Set environment variable:

```env
DEBUG=true
```

### Logs:

- FastAPI logs: Check terminal output
- Streamlit logs: Check browser console
- Database logs: Check PostgreSQL logs

## 📝 License

This project is created for HackRx 6.0 hackathon.

## 🤝 Support

For issues or questions:

1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Test with sample data first

---

**Built for HackRx 6.0** 🏆

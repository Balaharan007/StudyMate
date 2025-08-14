from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from simple_processor import SimpleDocumentProcessor

load_dotenv()

app = FastAPI(
    title="Document Intelligence Agent",
    description="AI-powered document analysis system with semantic processing",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Initialize document processor
doc_processor = SimpleDocumentProcessor()

# In-memory storage
documents_storage = {}
queries_storage = []

# Pydantic models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    return {
        "message": "🤖 Document Intelligence Agent",
        "version": "3.0.0",
        "status": "Production Ready",
        "features": [
            "✅ Document processing (PDF, DOCX, TXT)",
            "✅ File upload support", 
            "✅ Semantic search and retrieval",
            "✅ AI-powered answers with Gemini",
            "✅ HackRx API compliance",
            "✅ Insurance/Legal/HR/Compliance domains",
            "✅ Explainable clause-based reasoning"
        ],
        "endpoints": {
            "hackrx": "/hackrx/run",
            "upload": "/documents/upload-file",
            "upload-url": "/documents/upload-url",
            "test": "/test-upload",
            "query": "/query",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "hackrx-document-agent", 
        "version": "3.0.0",
        "ai_models": ["Gemini-2.0-Flash", "Simple-Text-Similarity"],
        "ready_for": "Production Use"
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    🎯 MAIN HACKRX ENDPOINT
    
    Processes documents and answers multiple questions with AI-powered analysis.
    Advanced document intelligence system with semantic processing.
    
    Features:
    - Multi-format document processing
    - Semantic chunk retrieval  
    - AI-powered answer generation
    - Explainable reasoning
    - Optimized for insurance/legal/HR domains
    """
    try:
        print(f"🔄 Processing HackRx request for document: {request.documents[:50]}...")
        
        # Check if document is already processed
        if request.documents not in documents_storage:
            print(f"📄 Processing new document...")
            result = doc_processor.process_document(request.documents)
            
            if not result["success"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document processing failed: {result['error']}"
                )
            
            # Store document in memory
            documents_storage[request.documents] = {
                "title": request.documents.split('/')[-1],
                "content": result["text"][:5000],  # Store sample content
                "chunks": result["chunks"],
                "document_id": result["document_id"]
            }
            document_id = result["document_id"]
            print(f"✅ Document processed: {result['chunks']} chunks created")
        else:
            document_id = documents_storage[request.documents]["document_id"]
            print(f"♻️  Using cached document")
        
        # Process each question with AI reasoning
        answers = []
        for i, question in enumerate(request.questions):
            print(f"🤔 Processing question {i+1}/{len(request.questions)}: {question[:80]}...")
            
            # Search for relevant chunks
            relevant_chunks = doc_processor.search_similar_chunks(
                query=question,
                document_id=document_id,
                top_k=5
            )
            
            # Generate answer using AI
            result = doc_processor.generate_answer(question, relevant_chunks)
            answer = result["answer"]
            
            # Store query for analytics
            queries_storage.append({
                "document_url": request.documents,
                "question": question,
                "answer": answer,
                "relevant_chunks": len(result["relevant_chunks"]),
                "reasoning": result.get("reasoning", "")
            })
            
            answers.append(answer)
            print(f"✅ Answer generated with {len(relevant_chunks)} relevant chunks")
        
        print(f"🎉 HackRx request completed successfully! Generated {len(answers)} answers.")
        return HackRxResponse(answers=answers)
    
    except Exception as e:
        print(f"❌ Error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/documents/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    token: str = Depends(verify_token)
):
    """📁 Upload and process document files (PDF, DOCX, TXT)"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = file.filename.lower().split('.')[-1]
        if file_ext not in ['pdf', 'docx', 'doc', 'txt']:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, DOCX, or TXT files."
            )
        
        # Read file content
        file_content = await file.read()
        file_identifier = f"file_{file.filename}_{len(file_content)}"
        
        # Check if already processed
        if file_identifier in documents_storage:
            return {
                "success": True,
                "message": "Document already processed",
                "document_id": documents_storage[file_identifier]["document_id"],
                "chunks": documents_storage[file_identifier]["chunks"]
            }
        
        # Process document
        result = doc_processor.process_document(
            source=file.filename,
            is_file_path=False,
            file_content=file_content,
            filename=file.filename
        )
        
        if not result["success"]:
            return {
                "success": False,
                "message": "Document processing failed",
                "error": result["error"]
            }
        
        # Store in memory
        documents_storage[file_identifier] = {
            "title": title or file.filename,
            "content": result["text"][:5000],
            "chunks": result["chunks"],
            "document_id": result["document_id"],
            "filename": file.filename
        }
        
        return {
            "success": True,
            "message": "Document processed successfully",
            "document_id": result["document_id"],
            "chunks": result["chunks"]
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": "File upload failed",
            "error": str(e)
        }

@app.post("/documents/upload-url")
async def upload_url(
    request: dict,
    token: str = Depends(verify_token)
):
    """🔗 Upload and process document from URL"""
    try:
        url = request.get("url")
        title = request.get("title", "")
        
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Check if already processed
        if url in documents_storage:
            return {
                "success": True,
                "message": "Document already processed",
                "document_id": documents_storage[url]["document_id"],
                "chunks": documents_storage[url]["chunks"]
            }
        
        # Process document from URL
        result = doc_processor.process_document(
            source=url,
            is_file_path=False
        )
        
        if not result["success"]:
            return {
                "success": False,
                "message": "Document processing failed",
                "error": result["error"]
            }
        
        # Store in memory
        documents_storage[url] = {
            "title": title or f"Document from URL",
            "content": result["text"][:5000],
            "chunks": result["chunks"],
            "document_id": result["document_id"],
            "url": url
        }
        
        return {
            "success": True,
            "message": "Document processed successfully from URL",
            "document_id": result["document_id"],
            "chunks": result["chunks"]
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": "URL upload failed",
            "error": str(e)
        }

@app.post("/query")
async def query_document(
    request: dict,
    token: str = Depends(verify_token)
):
    """🔍 Query specific documents with AI-powered analysis"""
    try:
        document_url = request.get("document_url")
        document_id = request.get("document_id")
        question = request.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Find document
        doc_info = None
        target_doc_id = None
        
        if document_url and document_url in documents_storage:
            doc_info = documents_storage[document_url]
            target_doc_id = doc_info["document_id"]
        elif document_id:
            for key, value in documents_storage.items():
                if value["document_id"] == document_id:
                    doc_info = value
                    target_doc_id = document_id
                    break
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Search and generate answer
        relevant_chunks = doc_processor.search_similar_chunks(
            query=question,
            document_id=target_doc_id,
            top_k=5
        )
        
        result = doc_processor.generate_answer(question, relevant_chunks)
        
        # Store query
        queries_storage.append({
            "document_id": target_doc_id,
            "question": question,
            "answer": result["answer"],
            "relevant_chunks": len(result["relevant_chunks"])
        })
        
        return {
            "answer": result["answer"],
            "relevant_chunks": result["relevant_chunks"],
            "reasoning": result.get("reasoning", ""),
            "document_title": doc_info["title"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(token: str = Depends(verify_token)):
    """📚 List all processed documents"""
    documents = []
    for key, doc in documents_storage.items():
        documents.append({
            "id": key,
            "url": key if key.startswith("http") else None,
            "title": doc["title"],
            "chunks": doc["chunks"],
            "document_id": doc["document_id"],
            "is_file_upload": not key.startswith("http"),
            "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
        })
    return documents

@app.get("/stats")
async def get_stats(token: str = Depends(verify_token)):
    """📊 System statistics and performance metrics"""
    total_docs = len(documents_storage)
    total_queries = len(queries_storage)
    
    file_uploads = sum(1 for key in documents_storage.keys() if not key.startswith("http"))
    url_documents = total_docs - file_uploads
    
    # Calculate average chunks per document
    total_chunks = sum(doc["chunks"] for doc in documents_storage.values())
    avg_chunks = total_chunks / max(total_docs, 1)
    
    return {
        "total_documents": total_docs,
        "total_queries": total_queries,
        "file_uploads": file_uploads,
        "url_documents": url_documents,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": round(avg_chunks, 1),
        "avg_queries_per_doc": round(total_queries / max(total_docs, 1), 2),
        "system_status": "Operational",
        "ai_models": ["Gemini-2.0-Flash-Exp", "Simple-Text-Similarity"],
        "compliance": "Production Ready"
    }

@app.get("/demo")
async def demo_endpoint():
    """🧪 Demo endpoint with sample data for testing"""
    return {
        "sample_request": {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "questions": [
                "What is the grace period for premium payment?",
                "Does this policy cover maternity expenses?",
                "What is the waiting period for cataract surgery?"
            ]
        },
        "curl_command": """
curl -X POST "http://localhost:8000/hackrx/run" \\
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"documents": "URL", "questions": ["QUESTION"]}'
        """,
        "expected_response": {
            "answers": ["Detailed answer based on document analysis..."]
        }
    }

@app.post("/test-upload")
async def test_upload(token: str = Depends(verify_token)):
    """Test endpoint to verify system functionality without external URLs"""
    return {
        "success": True,
        "message": "Test endpoint is working",
        "test": True
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Document Intelligence Agent...")
    print("🎯 Ready for intelligent document processing!")
    uvicorn.run(
        "main_final:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )

import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
import io

load_dotenv()

# Configuration
API_BASE_URL = f"http://localhost:{os.getenv('API_PORT', 8000)}"
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

st.set_page_config(
    page_title="Document Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}

/* Navigation radio buttons styling */
div[role="radiogroup"] label {
    font-size: 16px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    border-radius: 8px !important;
    background-color: transparent !important;
    border: 1px solid #e0e0e0 !important;
    transition: all 0.3s ease !important;
}

div[role="radiogroup"] label:hover {
    background-color: #e8f4fd !important;
    border-color: #1f77b4 !important;
}

div[role="radiogroup"] label[data-checked="true"] {
    background-color: #1f77b4 !important;
    color: white !important;
    border-color: #1f77b4 !important;
}

/* Main content styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Success/error message styling */
.stSuccess, .stError, .stInfo, .stWarning {
    border-radius: 8px !important;
    border-left: 4px solid !important;
}

/* Button styling */
.stButton > button {
    border-radius: 8px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

/* Metric styling */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #f8f9fa !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

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
        st.error(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response: {e.response.text}")
        return None

def main():
    st.title("Document Intelligence Agent")
    st.markdown("### AI-powered document analysis with file upload support")
    
    # Initialize session state variables
    if 'show_url_query' not in st.session_state:
        st.session_state.show_url_query = False
    if 'uploaded_doc_id' not in st.session_state:
        st.session_state.uploaded_doc_id = None
    
    # Check API connection
    if not test_api_connection():
        st.error("❌ **Cannot connect to FastAPI server!**")
        st.error(f"Please ensure the server is running at: {API_BASE_URL}")
        st.info("💡 **To start the server:**")
        st.code("start_server.bat", language="bash")
        st.info("Or run manually:")
        st.code("python -m uvicorn main_final:app --host 0.0.0.0 --port 8000", language="bash")
        st.stop()
    else:
        st.success(f"✅ Connected to FastAPI server at {API_BASE_URL}")
    
    # Sidebar with improved styling
    st.sidebar.markdown("# 🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Enhanced page selection with larger text
    page = st.sidebar.radio(
        "",
        ["🧪 HackRx API Test", "📁 File Upload", "🔗 URL Upload", "🔍 Query Documents", "📚 Document Management", "📊 System Stats"],
        format_func=lambda x: f"### {x}"
    )
    
    if page == "🧪 HackRx API Test":
        hackrx_api_test()
    elif page == "📁 File Upload":
        file_upload_page()
    elif page == "🔗 URL Upload":
        url_upload_page()
    elif page == "🔍 Query Documents":
        query_documents_page()
    elif page == "📚 Document Management":
        document_management_page()
    elif page == "📊 System Stats":
        system_stats_page()

def hackrx_api_test():
    """HackRx API test page"""
    st.header("🧪 HackRx API Test")
    st.markdown("Test the main `/hackrx/run` endpoint with the official sample data")
    
    # Demo mode option
    with st.expander("🚀 Demo Mode (Recommended)", expanded=True):
        st.info("Try the system with a sample policy document using a working URL")
        if st.button("🎯 Run HackRx Demo", type="primary"):
            # Use a working public PDF URL for demo
            demo_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            
            demo_questions = [
                "What is this document about?",
                "What type of document is this?",
                "What is the main content?"
            ]
            
            request_data = {
                "documents": demo_url,
                "questions": demo_questions
            }
            
            with st.spinner("🔍 Processing demo queries..."):
                result = make_api_request("/hackrx/run", request_data, "POST")
            
            if result:
                st.success("✅ Demo completed successfully!")
                st.subheader("📋 Demo Results")
                
                if "answers" in result:
                    for i, (question, answer) in enumerate(zip(demo_questions, result["answers"])):
                        with st.expander(f"Q{i+1}: {question}"):
                            st.write("**Question:**")
                            st.write(question)
                            st.write("**Answer:**")
                            st.write(answer)
            else:
                st.error("❌ Demo failed - check backend logs")
                st.info("💡 Try using the manual URL input below with a working document URL")
    
    st.divider()
    
    # Default sample data
    default_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    default_questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    st.subheader("📋 Request Configuration")
    
    # Document URL input
    document_url = st.text_input(
        "Document URL:",
        value=default_url,
        help="Enter the URL of the document to analyze"
    )
    
    # Questions selection
    st.subheader("❓ Questions")
    selected_questions = st.multiselect(
        "Select questions to process:",
        default_questions,
        default=default_questions[:5],
        help="Select which questions to send to the API"
    )
    
    # Add custom question
    custom_question = st.text_area(
        "Add custom question:",
        placeholder="Enter your own question about the document...",
        height=60
    )
    
    if custom_question.strip():
        selected_questions.append(custom_question.strip())
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Run HackRx API", type="primary"):
            if not document_url:
                st.error("Please enter a document URL")
                return
            
            if not selected_questions:
                st.error("Please select at least one question")
                return
            
            # Prepare request
            request_data = {
                "documents": document_url,
                "questions": selected_questions
            }
            
            # Show request data
            with st.expander("📤 Request Data"):
                st.json(request_data)
            
            # Make API call
            with st.spinner(f"Processing document and generating {len(selected_questions)} answers..."):
                result = make_api_request("/hackrx/run", request_data, "POST")
            
            if result:
                st.success("✅ Successfully processed!")
                
                # Display results
                st.subheader("📋 Results")
                
                if "answers" in result:
                    for i, (question, answer) in enumerate(zip(selected_questions, result["answers"])):
                        with st.expander(f"Q{i+1}: {question[:80]}..."):
                            st.write("**Question:**")
                            st.write(question)
                            st.write("**Answer:**")
                            st.write(answer)
                
                # Show raw response
                with st.expander("🔍 Raw API Response"):
                    st.json(result)
            else:
                st.error("❌ API request failed. Check the console for details.")
                st.markdown("**Sample working URLs to test:**")
                st.code("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
                st.code("https://file-examples.com/storage/fe68c1b20e66dbf3c9d8ba3/2017/10/file_example_PDF_1MB.pdf")
    
    with col2:
        if st.button("🔄 Reset"):
            st.rerun()

def file_upload_page():
    """File upload page"""
    st.header("📁 File Upload & Processing")
    st.markdown("Upload your documents (PDF, DOCX, TXT) directly")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Document File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file:",
            type=["pdf", "docx", "doc", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Title input
        doc_title = st.text_input(
            "Document Title (optional):",
            placeholder="Enter a title for the document"
        )
        
        if uploaded_file is not None:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            st.write(f"**File type:** {uploaded_file.type}")
            
            if st.button("📤 Upload & Process Document", type="primary"):
                with st.spinner("Uploading and processing document..."):
                    # Prepare files for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"title": doc_title} if doc_title else {}
                    
                    result = make_api_request("/documents/upload-file", data, "POST", files)
                
                if result:
                    if result.get("success"):
                        st.success(f"✅ {result['message']}")
                        
                        # Show document details in a nice info box
                        with st.container():
                            st.markdown("### 📋 Document Processed Successfully!")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Document ID", result.get('document_id', 'N/A'))
                            with col2:
                                st.metric("Chunks Created", result.get('chunks', 0))
                            with col3:
                                st.metric("Status", "✅ Ready")
                        
                        # Option to query immediately
                        st.divider()
                        st.subheader("🔍 Ask a Question About This Document")
                        
                        # Sample questions
                        st.markdown("**💡 Try these sample questions:**")
                        sample_questions = [
                            "What is this document about?",
                            "What are the key points?",
                            "What are the main requirements?",
                            "What are the terms and conditions?"
                        ]
                        
                        selected_sample = st.selectbox("Or choose a sample question:", [""] + sample_questions)
                        
                        question = st.text_area(
                            "Enter your question:",
                            value=selected_sample if selected_sample else "",
                            height=100,
                            placeholder="What would you like to know about this document?"
                        )
                        if question and st.button("🔍 Ask Question", type="primary"):
                            query_data = {
                                "document_id": result.get('document_id'),
                                "question": question
                            }
                            with st.spinner("🔍 Analyzing document and generating answer..."):
                                query_result = make_api_request("/query", query_data, "POST")
                            
                            if query_result:
                                st.success("✅ Answer generated!")
                                
                                # Display answer prominently
                                st.markdown("### 💬 Answer")
                                st.markdown(f"**{query_result.get('answer')}**")
                                
                                if query_result.get("reasoning"):
                                    st.info(f"🧠 **Reasoning:** {query_result['reasoning']}")
                                
                                if query_result.get("relevant_chunks"):
                                    with st.expander("📚 Relevant Document Sections", expanded=True):
                                        for i, chunk in enumerate(query_result["relevant_chunks"]):
                                            st.write(f"**📄 Section {i+1}** (Relevance: {chunk.get('score', 0):.3f})")
                                            st.write(f"📝 {chunk.get('text', '')}")
                                            if i < len(query_result["relevant_chunks"]) - 1:
                                                st.divider()
                            else:
                                st.error("❌ Failed to get answer from the document.")
                    else:
                        st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.subheader("📊 Upload Status")
        if st.button("🔄 Refresh"):
            stats = make_api_request("/stats")
            if stats:
                st.metric("Total Documents", stats.get("total_documents", 0))
                st.metric("File Uploads", stats.get("file_uploads", 0))
                st.metric("URL Documents", stats.get("url_documents", 0))

def url_upload_page():
    """URL upload page"""
    st.header("🔗 URL Upload & Processing")
    st.markdown("Process documents from URLs")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Document from URL")
        
        # Demo section
        with st.expander("🧪 Try Demo Mode (With Working URL)", expanded=False):
            st.info("Click below to test the system with a working demo document URL")
            if st.button("🚀 Load Demo Document", type="primary"):
                # Use a working demo URL
                demo_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
                
                request_data = {
                    "url": demo_url,
                    "title": "Demo PDF Document"
                }
                
                with st.spinner("Loading demo document..."):
                    demo_response = make_api_request("/documents/upload-url", request_data, "POST")
                
                if demo_response and demo_response.get("success"):
                    st.success(f"✅ {demo_response['message']}")
                    st.info(f"Document ID: {demo_response.get('document_id')}")
                    st.info(f"Number of chunks: {demo_response.get('chunks')}")
                    
                    # Store the document ID for querying
                    st.session_state.uploaded_doc_id = demo_response.get('document_id')
                    st.session_state.show_url_query = True
                    st.success("💡 Demo document loaded! You can now ask questions below.")
                else:
                    st.error("❌ Failed to load demo document")
                    st.info("💡 Try entering a URL manually below")
        
        st.divider()
        
        # URL input
        doc_url = st.text_input(
            "Document URL:",
            placeholder="https://example.com/document.pdf",
            help="Enter the URL of the PDF, DOCX, or TXT document"
        )
        
        # Title input
        doc_title = st.text_input(
            "Document Title (optional):",
            placeholder="Enter a title for the document"
        )
        
        if st.button("📤 Upload & Process Document", type="primary"):
            if not doc_url:
                st.error("Please enter a document URL")
                return
            
            request_data = {
                "url": doc_url,
                "title": doc_title
            }
            
            with st.spinner("Downloading and processing document..."):
                result = make_api_request("/documents/upload-url", request_data, "POST")
            
            if result:
                if result.get("success"):
                    st.success(f"✅ {result['message']}")
                    st.info(f"Document ID: {result.get('document_id')}")
                    st.info(f"Number of chunks: {result.get('chunks')}")
                    
                    # Store the document ID in session state for querying
                    st.session_state.uploaded_doc_id = result.get('document_id')
                    st.session_state.show_url_query = True
                else:
                    error_msg = result.get('error', 'Unknown error')
                    detail_msg = result.get('detail', '')
                    st.error(f"❌ Processing failed: {error_msg}")
                    if detail_msg:
                        st.error(f"Details: {detail_msg}")
                    
                    # Show helpful tips for common errors
                    if "download" in error_msg.lower():
                        st.info("💡 Tip: Make sure the URL is publicly accessible and points directly to a document")
                    elif "format" in error_msg.lower():
                        st.info("💡 Tip: Supported formats are PDF, DOCX, DOC, and TXT files")
                    
                    st.markdown("**Sample working URLs to test:**")
                    st.code("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
                    st.code("https://file-examples.com/storage/fe68c1b20e66dbf3c9d8ba3/2017/10/file_example_PDF_1MB.pdf")
        
        # Query functionality - Show after successful upload
        if st.session_state.get('show_url_query') and st.session_state.get('uploaded_doc_id'):
            st.markdown("---")
            st.subheader("🔍 Query This Document")
            st.markdown("Now you can ask questions about the uploaded document:")
            
            # Quick question buttons
            st.write("**Quick Questions:**")
            quick_questions = [
                "What is the purpose of this document?",
                "What are the main policies?", 
                "Does this policy apply to all employees?",
                "What is the effective date?",
                "Are there medical benefits mentioned?"
            ]
            
            # Display question buttons in rows
            cols = st.columns(3)
            selected_quick_question = None
            
            for i, question in enumerate(quick_questions):
                with cols[i % 3]:
                    if st.button(f"❓ {question[:20]}...", key=f"quick_q_{i}", help=question):
                        selected_quick_question = question
            
            # Custom question input
            st.write("**Custom Question:**")
            custom_question = st.text_area(
                "Enter your own question about the document:",
                placeholder="Type your question here...",
                height=80,
                key="url_custom_question"
            )
            
            # Query button
            query_question = selected_quick_question or custom_question.strip()
            
            if st.button("🚀 Ask Question", type="primary", disabled=not query_question):
                if query_question:
                    query_data = {
                        "question": query_question,
                        "document_id": st.session_state.uploaded_doc_id
                    }
                    
                    with st.spinner("🔍 Analyzing document and generating answer..."):
                        query_result = make_api_request("/query", query_data, "POST")
                    
                    if query_result:
                        st.success("✅ Answer generated!")
                        
                        # Display answer prominently
                        st.markdown("### 💬 Answer")
                        st.markdown(f"**Question:** {query_question}")
                        st.markdown(f"**Answer:** {query_result.get('answer')}")
                        
                        # Show additional details if available
                        if query_result.get('source_chunks'):
                            with st.expander("📚 Source Information"):
                                st.write(f"**Sources:** {len(query_result['source_chunks'])} relevant chunks found")
                                for i, chunk in enumerate(query_result['source_chunks'][:3]):
                                    st.write(f"**Chunk {i+1}:** {chunk[:200]}...")
                    else:
                        st.error("❌ Failed to get answer. Please try again.")
            
            # Reset button
            if st.button("🔄 Upload Another Document"):
                st.session_state.show_url_query = False
                st.session_state.uploaded_doc_id = None
                st.rerun()
    
    with col2:
        st.subheader("📋 Recent URLs")
        documents = make_api_request("/documents")
        if documents:
            url_docs = [doc for doc in documents if (doc.get("url") or "").startswith("http")]
            for doc in url_docs[-3:]:  # Show last 3 URL documents
                st.write(f"📄 {doc.get('title', 'Untitled')[:30]}...")
                st.caption(f"Chunks: {doc.get('chunks', 0)}")

def query_documents_page():
    """Query documents page"""
    st.header("🔍 Query Documents")
    st.markdown("Ask questions about your processed documents")
    
    # Get list of documents
    documents = make_api_request("/documents")
    
    if not documents:
        st.warning("No documents found. Please upload a document first.")
        st.markdown("Use the **File Upload** or **URL Upload** pages to add documents.")
        return
    
    # Document selection
    doc_options = {}
    for doc in documents:
        url = doc.get("url") or ""
        doc_type = "📁 File" if not url.startswith("http") else "🔗 URL"
        doc_options[doc['id']] = f"{doc_type}: {doc['title']} ({doc.get('chunks', 0)} chunks)"
    
    selected_doc_id = st.selectbox(
        "Select Document:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    # Quick question buttons
    st.subheader("🎯 Quick Questions")
    quick_questions = [
        "What is the main purpose of this document?",
        "What are the key terms and conditions?",
        "What are the eligibility criteria?",
        "What are the exclusions or limitations?",
        "What is the process for claims or applications?"
    ]
    
    cols = st.columns(len(quick_questions))
    selected_quick = None
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(f"❓ {q[:20]}...", key=f"quick_{i}"):
                selected_quick = q
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        value=selected_quick if selected_quick else "",
        height=100,
        placeholder="What specific information are you looking for in this document?"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔍 Ask Question", type="primary"):
            if not question.strip():
                st.error("Please enter a question")
                return
            
            # Get document details
            selected_doc = next((doc for doc in documents if doc['id'] == selected_doc_id), None)
            
            if not selected_doc:
                st.error("Document not found")
                return
            
            request_data = {
                "document_id": selected_doc.get('document_id'),
                "question": question
            }
            
            st.info(f"🔍 Querying document: **{selected_doc['title']}**")
            
            with st.spinner("Searching document and generating answer..."):
                result = make_api_request("/query", request_data, "POST")
            
            if result:
                st.success("✅ Answer generated successfully!")
                st.subheader("💬 Answer")
                st.markdown(f"**{result['answer']}**")
                
                if result.get("reasoning"):
                    st.info(f"🧠 **Reasoning:** {result['reasoning']}")
                
                if "relevant_chunks" in result and result["relevant_chunks"]:
                    with st.expander("📚 Relevant Document Sections", expanded=True):
                        for i, chunk in enumerate(result["relevant_chunks"]):
                            st.write(f"**📄 Section {i+1}** (Relevance Score: {chunk.get('score', 0):.3f})")
                            st.write(f"📝 {chunk['text']}")
                            if i < len(result["relevant_chunks"]) - 1:
                                st.divider()
            else:
                st.error("❌ Failed to get answer. Please check the API connection.")

def document_management_page():
    """Document management page"""
    st.header("📚 Document Management")
    st.markdown("Manage your processed documents and view query history")
    
    # Get documents
    documents = make_api_request("/documents")
    
    if not documents:
        st.info("No documents found.")
        return
    
    st.subheader(f"📋 All Documents ({len(documents)})")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📄 Document List", "📊 Document Details"])
    
    with tab1:
        for doc in documents:
            url = doc.get("url") or ""
            doc_type = "📁 File Upload" if not (url or "").startswith("http") else "🔗 URL Document"
            
            with st.expander(f"{doc_type}: {doc['title']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**ID:** {doc['id']}")
                    st.write(f"**Title:** {doc['title']}")
                    st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
                    st.write(f"**Chunks:** {doc.get('chunks', 0)}")
                    st.write(f"**Document ID:** {doc.get('document_id', 'N/A')}")
                
                with col2:
                    if st.button(f"📊 View Queries", key=f"queries_{doc['id']}"):
                        queries = make_api_request(f"/documents/{doc['id']}/queries")
                        if queries:
                            st.subheader("Previous Queries")
                            for query in queries[-5:]:  # Show last 5
                                st.write(f"**Q:** {query['question'][:100]}...")
                                st.write(f"**A:** {query['answer'][:200]}...")
                                st.caption(f"Asked: {query['created_at']}")
                                st.write("---")
                        else:
                            st.info("No queries found for this document.")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}"):
                        if st.button(f"Confirm Delete", key=f"confirm_{doc['id']}"):
                            result = make_api_request(f"/documents/{doc['id']}", method="DELETE")
                            if result:
                                st.success("Document deleted!")
                                st.rerun()
    
    with tab2:
        # Document statistics
        stats = make_api_request("/stats")
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Total Documents", stats.get("total_documents", 0))
            with col2:
                st.metric("❓ Total Queries", stats.get("total_queries", 0))
            with col3:
                st.metric("📁 File Uploads", stats.get("file_uploads", 0))
            with col4:
                st.metric("🔗 URL Documents", stats.get("url_documents", 0))
            
            st.metric("📊 Avg Queries/Doc", stats.get("avg_queries_per_doc", 0))

def system_stats_page():
    """System statistics page"""
    st.header("📊 System Statistics")
    st.markdown("Overview of system usage and performance")
    
    stats = make_api_request("/stats")
    documents = make_api_request("/documents")
    
    if stats and documents:
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("❓ Total Queries", stats.get("total_queries", 0))
        with col3:
            st.metric("📁 File Uploads", stats.get("file_uploads", 0))
        with col4:
            st.metric("🔗 URL Documents", stats.get("url_documents", 0))
        
        # Additional metrics
        st.subheader("📈 Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Avg Queries per Document", stats.get("avg_queries_per_doc", 0))
            
            # Document type breakdown
            if documents:
                file_count = sum(1 for doc in documents if not (doc.get("url") or "").startswith("http"))
                url_count = len(documents) - file_count
                
                st.write("**Document Types:**")
                st.write(f"- File Uploads: {file_count}")
                st.write(f"- URL Documents: {url_count}")
        
        with col2:
            if documents:
                # Chunk statistics
                total_chunks = sum(doc.get("chunks", 0) for doc in documents)
                avg_chunks = total_chunks / len(documents) if documents else 0
                
                st.metric("🧩 Total Chunks", total_chunks)
                st.metric("📏 Avg Chunks per Doc", round(avg_chunks, 1))
        
        # Recent documents
        st.subheader("📋 Recent Documents")
        recent_docs = sorted(documents, key=lambda x: x.get("created_at", ""), reverse=True)[:5]
        
        for doc in recent_docs:
            doc_type = "📁" if not (doc.get("url") or "").startswith("http") else "🔗"
            st.write(f"{doc_type} **{doc['title']}** - {doc.get('chunks', 0)} chunks - {doc.get('created_at', '')}")

if __name__ == "__main__":
    main()

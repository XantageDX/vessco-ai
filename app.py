# # app.py
# import os
# import streamlit as st
# import tempfile
# import shutil
# from dotenv import load_dotenv
# import datetime
# import json
# import traceback

# # Import our modules
# from embeddings import load_and_split_documents, load_and_split_documents_hierarchically
# from embeddings import initialize_vector_db, get_retriever, add_to_vector_db
# from bedrock_llm import get_bedrock_llm
# from rag_chain import create_rag_chain, create_enhanced_rag_chain
# from archive_manager import delete_document, get_archive_documents
# from portfolio_manager import PortfolioManager
# from product_matcher import ProductMatcher
# from spec_extractor import SpecificationExtractor
# from debug_tools import debug_trace, monkey_patch_llm

# # Enable debug mode
# DEBUG_MODE = os.environ.get("VESSCO_DEBUG", "false").lower() == "true"

# # Load environment variables
# load_dotenv()

# # Set page configuration
# st.set_page_config(
#     page_title="Vessco AI Assistant",
#     page_icon="💬",
#     layout="wide"
# )

# # Initialize session state variables
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "chat_chain" not in st.session_state:
#     st.session_state.chat_chain = None

# if "db_initialized" not in st.session_state:
#     st.session_state.db_initialized = False

# if "use_hierarchical" not in st.session_state:
#     st.session_state.use_hierarchical = True

# if "portfolio_manager" not in st.session_state:
#     st.session_state.portfolio_manager = PortfolioManager()

# # App title
# st.title("Vessco AI Assistant")

# # Sidebar for configuration and document upload
# with st.sidebar:
#     st.header("Configuration")
    
#     # Model selection
#     model_id = st.selectbox(
#         "Select LLM Model",
#         ["us.meta.llama3-3-70b-instruct-v1:0", 
#          #"anthropic.claude-3-sonnet-20240229-v1:0",
#          "us.meta.llama3-1-8b-instruct-v1:0"],
#         index=0
#     )
    
#     # Processing method selection
#     st.session_state.use_hierarchical = st.checkbox(
#         "Use Hierarchical Document Processing", 
#         value=st.session_state.use_hierarchical,
#         help="Process documents by chapters/sections instead of arbitrary chunks"
#     )
    
#     # Debug mode toggle (hidden in an expander to not confuse regular users)
#     with st.expander("Advanced Settings"):
#         prev_debug = DEBUG_MODE
#         DEBUG_MODE = st.checkbox(
#             "Enable Debug Mode", 
#             value=DEBUG_MODE,
#             help="Log detailed information for troubleshooting"
#         )
#         if DEBUG_MODE != prev_debug:
#             os.environ["VESSCO_DEBUG"] = "true" if DEBUG_MODE else "false"
            
#         if DEBUG_MODE:
#             st.info("Debug mode is enabled. Detailed error information and logs will be shown.")
#             if st.button("Clear Debug Logs"):
#                 try:
#                     if os.path.exists("vessco_debug.log"):
#                         os.remove("vessco_debug.log")
#                     if os.path.exists("llm_logs"):
#                         for file in os.listdir("llm_logs"):
#                             os.remove(os.path.join("llm_logs", file))
#                     st.success("Debug logs cleared")
#                 except Exception as e:
#                     st.error(f"Error clearing logs: {str(e)}")
    
#     # Document Upload section
#     st.header("Document Upload")
    
#     # File uploader with drag and drop
#     st.write("Upload documents for RAG")
#     uploaded_files = st.file_uploader("Drag and drop files here", 
#                                      accept_multiple_files=True,
#                                      type=["pdf", "txt", "docx"],
#                                      key="temp_uploader",
#                                      label_visibility="collapsed")
    
#     # Process files
#     col1, col2 = st.columns(2)
#     with col1:
#         process_temp = st.button("Process Only", key="process_temp")
#     with col2:
#         archive_upload = st.button("Process & Archive", key="archive_upload")
    
#     if uploaded_files and (process_temp or archive_upload):
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Create a temporary directory to store the uploaded files
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save uploaded files to the temporary directory
#             for i, file in enumerate(uploaded_files):
#                 progress = (i) / len(uploaded_files) * 0.3
#                 progress_bar.progress(progress)
#                 status_text.text(f"Saving file {i+1}/{len(uploaded_files)}: {file.name}")
                
#                 file_path = os.path.join(temp_dir, file.name)
#                 with open(file_path, "wb") as f:
#                     f.write(file.getvalue())
            
#             # Process the documents
#             status_text.text("Processing documents...")
#             progress_bar.progress(0.4)
            
#             # Define a callback function to update status
#             def update_status(message):
#                 status_text.text(message)
            
#             # Get LLM for document processing if using hierarchical approach
#             llm = None
#             if st.session_state.use_hierarchical:
#                 llm = get_bedrock_llm(model_id)
#                 # Apply debug logging in debug mode
#                 if DEBUG_MODE:
#                     llm = monkey_patch_llm(llm)
                
#             # Load and split documents with status updates
#             try:
#                 if st.session_state.use_hierarchical and llm:
#                     chunks = load_and_split_documents_hierarchically(temp_dir, llm, status_callback=update_status)
#                 else:
#                     chunks = load_and_split_documents(temp_dir, status_callback=update_status)
#             except Exception as e:
#                 error_details = traceback.format_exc()
#                 status_text.error(f"Error processing documents: {str(e)}")
#                 if DEBUG_MODE:
#                     st.error(f"Detailed error: {error_details}")
#                 st.stop()
                
#             progress_bar.progress(0.6)
            
#             # Archive the files if requested
#             if archive_upload:
#                 status_text.text("Archiving documents...")
#                 progress_bar.progress(0.7)
                
#                 # Create archive directory if it doesn't exist
#                 archive_dir = "./uploaded_documents"
#                 os.makedirs(archive_dir, exist_ok=True)
                
#                 # Copy files to archive with timestamp
#                 for file in uploaded_files:
#                     # Add timestamp to avoid overwriting files with same name
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                     file_name_parts = os.path.splitext(file.name)
#                     archive_filename = f"{file_name_parts[0]}_{timestamp}{file_name_parts[1]}"
                    
#                     # Save to archive
#                     archive_path = os.path.join(archive_dir, archive_filename)
#                     with open(archive_path, "wb") as f:
#                         f.write(file.getvalue())
            
#             # Initialize or update vector database
#             status_text.text("Updating vector database...")
            
#             # Add detailed status updates for vector database process
#             total_chunks = len(chunks)
#             batch_size = max(1, total_chunks // 10)  # Create 10 update points
            
#             if not st.session_state.db_initialized:
#                 # Initialize with detailed progress updates
#                 for i in range(0, total_chunks, batch_size):
#                     end_idx = min(i + batch_size, total_chunks)
#                     curr_batch = chunks[i:end_idx]
#                     if i == 0:  # First batch initializes the database
#                         initialize_vector_db(curr_batch)
#                         st.session_state.db_initialized = True
#                     else:  # Subsequent batches are added to the database
#                         add_to_vector_db(curr_batch)
#                     # Update progress
#                     progress_percent = 0.8 + (0.2 * (end_idx / total_chunks))
#                     progress_bar.progress(progress_percent)
#                     status_text.text(f"Updating vector database... ({end_idx}/{total_chunks} chunks)")
#             else:
#                 # Add to existing database with progress updates
#                 for i in range(0, total_chunks, batch_size):
#                     end_idx = min(i + batch_size, total_chunks)
#                     curr_batch = chunks[i:end_idx]
#                     add_to_vector_db(curr_batch)
#                     # Update progress
#                     progress_percent = 0.8 + (0.2 * (end_idx / total_chunks))
#                     progress_bar.progress(progress_percent)
#                     status_text.text(f"Updating vector database... ({end_idx}/{total_chunks} chunks)")
            
#             progress_bar.progress(1.0)
#             if archive_upload:
#                 status_text.text("Processing and archiving complete!")
#             else:
#                 status_text.text("Processing complete!")
            
#             # Initialize or reinitialize the RAG chain
#             try:
#                 if st.session_state.use_hierarchical:
#                     st.session_state.chat_chain = create_enhanced_rag_chain(model_id=model_id)
#                     # Apply debug logging in debug mode
#                     if DEBUG_MODE and "llm" in st.session_state.chat_chain:
#                         st.session_state.chat_chain["llm"] = monkey_patch_llm(st.session_state.chat_chain["llm"])
#                 else:
#                     st.session_state.chat_chain = create_rag_chain(model_id=model_id)
#             except Exception as e:
#                 error_details = traceback.format_exc()
#                 status_text.error(f"Error initializing chat chain: {str(e)}")
#                 if DEBUG_MODE:
#                     st.error(f"Detailed error: {error_details}")
#                 st.stop()
            
#             if archive_upload:
#                 st.success(f"Successfully processed and archived {len(uploaded_files)} documents ({len(chunks)} chunks)")
#             else:
#                 st.success(f"Successfully processed {len(chunks)} document chunks")
    
#     # Document Archive section
#     st.header("Document Archive")
    
#     # Display archived documents and allow deletion
#     try:
#         archived_docs = get_archive_documents()
#         if archived_docs:
#             st.write("Manage Archived Documents")
            
#             for doc in archived_docs:
#                 col1, col2 = st.columns([3, 1])
#                 with col1:
#                     st.text(doc)
#                 with col2:
#                     if st.button("Delete", key=f"del_{doc}"):
#                         result = delete_document(doc)
#                         if result:
#                             st.success(f"Deleted {doc}")
#                             st.rerun()
#                         else:
#                             st.error(f"Failed to delete {doc}")
#         else:
#             st.info("No documents in archive. Use 'Process & Archive' to add documents.")
#     except Exception as e:
#         st.error(f"Error listing archived documents: {str(e)}")

#     # Initialize button
#     if not st.session_state.db_initialized:
#         init_col1, init_col2 = st.columns(2)
#         with init_col1:
#             if st.button("Initialize Empty Database"):
#                 status = st.empty()
#                 status.text("Initializing empty vector database...")
                
#                 # Create directory if it doesn't exist
#                 os.makedirs("./chroma_db", exist_ok=True)
                
#                 # Initialize the chat chain
#                 try:
#                     if st.session_state.use_hierarchical:
#                         st.session_state.chat_chain = create_enhanced_rag_chain(model_id=model_id)
#                     else:
#                         st.session_state.chat_chain = create_rag_chain(model_id=model_id)
#                     st.session_state.db_initialized = True
#                     status.text("Initialization complete!")
#                     st.success("Empty database initialized. You can now start chatting or add documents.")
#                 except Exception as e:
#                     st.error(f"Initialization failed: {str(e)}")
    
#     # Product Portfolio section
#     st.header("Product Portfolio")
    
#     # Display companies from portfolio
#     portfolio_manager = st.session_state.portfolio_manager
#     companies = portfolio_manager.get_companies()
    
#     if companies:
#         selected_company = st.selectbox("Select Company", companies)
#         if selected_company:
#             products = portfolio_manager.get_products_by_company(selected_company)
#             st.write(f"Products for {selected_company}:")
#             for product_type in products:
#                 for product_name, description in product_type.items():
#                     if product_name != "Use Case":
#                         st.write(f"- {product_name}")
#                         with st.expander("Details"):
#                             st.write(description)
#                             st.write(f"**Use Case:** {product_type.get('Use Case', 'Not specified')}")
#     else:
#         st.info("No products in portfolio. Add your first product below.")
    
#     # Add product form
#     with st.expander("Add New Product"):
#         new_company = st.text_input("Company Name")
#         new_product = st.text_input("Product Name")
#         new_description = st.text_area("Product Description")
#         new_use_case = st.text_area("Use Case")
        
#         if st.button("Add Product") and new_company and new_product and new_description:
#             if portfolio_manager.add_product(new_company, new_product, new_description, new_use_case):
#                 st.success(f"Added {new_product} to {new_company}")
#                 # Refresh portfolio 
#                 portfolio_manager.reload_portfolio()
#                 st.rerun()
#             else:
#                 st.error("Failed to add product")

# # Main content area - Chat interface
# # Create tabs for different views
# tab1, tab2 = st.tabs(["Chat", "Analysis"])

# with tab1:
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask me anything about your documents..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
            
#             if not st.session_state.db_initialized:
#                 response = "Please initialize the database or upload documents first."
#             elif not st.session_state.chat_chain:
#                 response = "Chat system is not initialized. Please upload documents or initialize the database."
#             else:
#                 with st.spinner("Thinking..."):
#                     try:
#                         # Get response from RAG chain using the updated invoke method
#                         response_data = st.session_state.chat_chain["invoke"]({"question": prompt})
                        
#                         # Store full response data in session state for the Analysis tab
#                         if "last_response_data" not in st.session_state:
#                             st.session_state.last_response_data = {}
#                         st.session_state.last_response_data = response_data
                        
#                         # Display text response
#                         if isinstance(response_data, dict) and "answer" in response_data:
#                             response = response_data["answer"]
#                         else:
#                             response = response_data
#                     except Exception as e:
#                         error_details = traceback.format_exc()
#                         response = f"Error generating response: {str(e)}"
#                         if DEBUG_MODE:
#                             response += f"\n\nDetailed error:\n```\n{error_details}\n```"
            
#             message_placeholder.markdown(response)
        
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

# with tab2:
#     st.header("Document and Product Analysis")
    
#     # Display last response data if available
#     if "last_response_data" in st.session_state and st.session_state.last_response_data:
#         data = st.session_state.last_response_data
        
#         # Display extracted specifications
#         if "specifications" in data and data["specifications"]:
#             st.subheader("Extracted Specifications")
            
#             # Create a formatted table for specifications
#             spec_data = []
#             for spec in data["specifications"]:
#                 # Collect unique page numbers across all specifications
#                 page_numbers = spec.get("page_numbers", [])
                
#                 # Create displayable source information
#                 source_doc = spec.get("source_document", "Unknown")
#                 page_count = len(page_numbers) if page_numbers else 0
#                 page_refs = ", ".join(page_numbers) if page_numbers else "N/A"
                
#                 spec_data.append({
#                     "Parameter": spec.get("parameter", ""),
#                     "Value": spec.get("value", ""),
#                     "Unit": spec.get("unit", ""),
#                     "Constraint": spec.get("constraint_type", ""),
#                     "Source": source_doc,
#                     "Pages": page_count,
#                     "Page Refs": page_refs
#                 })
            
#             if spec_data:
#                 st.table(spec_data)
#         else:
#             st.info("No specifications extracted from the last query.")
            
#         # Display product matches
#         if "matches" in data and data["matches"]:
#             st.subheader("Product Matches")
            
#             # Create tabs for each product
#             product_tabs = st.tabs([f"{m['company']} - {m['product']} ({m['score']}%)" 
#                                     for m in data["matches"][:5]])
            
#             for i, tab in enumerate(product_tabs):
#                 with tab:
#                     match = data["matches"][i]
#                     st.markdown(f"### {match['product']}")
#                     st.markdown(f"**Company:** {match['company']}")
#                     st.markdown(f"**Match Score:** {match['score']}%")
#                     st.markdown("**Reasoning:**")
#                     st.markdown(match['reasoning'])
                    
#                     if "description" in match:
#                         st.markdown("**Product Description:**")
#                         st.markdown(match['description'])
                    
#                     if "use_case" in match:
#                         st.markdown("**Use Case:**")
#                         st.markdown(match['use_case'])
#         else:
#             st.info("No product matches found for the last query.")
#     else:
#         st.info("Ask a question in the Chat tab to see analysis here.")

# app.py
import os
import streamlit as st
import tempfile
import shutil
from dotenv import load_dotenv
import datetime
import json
import traceback
import pandas as pd

# Import our modules
from embeddings import load_and_split_documents, load_and_split_documents_hierarchically
from embeddings import initialize_vector_db, get_retriever, add_to_vector_db
from bedrock_llm import get_bedrock_llm
from rag_chain import create_rag_chain, create_enhanced_rag_chain
from archive_manager import delete_document, get_archive_documents
from portfolio_manager import PortfolioManager
from product_matcher import ProductMatcher
from spec_extractor import SpecificationExtractor
from debug_tools import debug_trace, monkey_patch_llm
from specification_manager import SpecificationManager

# Enable debug mode
DEBUG_MODE = os.environ.get("VESSCO_DEBUG", "false").lower() == "true"

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Vessco AI Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

if "use_hierarchical" not in st.session_state:
    st.session_state.use_hierarchical = True

if "portfolio_manager" not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()

# Initialize specification manager if not already there
if "specification_manager" not in st.session_state:
    st.session_state.specification_manager = None

if "current_specifications" not in st.session_state:
    st.session_state.current_specifications = []

if "product_matches" not in st.session_state:
    st.session_state.product_matches = []

# App title
st.title("Vessco AI Assistant")

# Sidebar for configuration and document upload
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_id = st.selectbox(
        "Select LLM Model",
        ["us.meta.llama3-3-70b-instruct-v1:0", 
         #"anthropic.claude-3-sonnet-20240229-v1:0",
         "us.meta.llama3-1-8b-instruct-v1:0"],
        index=0
    )
    
    # Processing method selection
    st.session_state.use_hierarchical = st.checkbox(
        "Use Hierarchical Document Processing", 
        value=st.session_state.use_hierarchical,
        help="Process documents by chapters/sections instead of arbitrary chunks"
    )
    
    # Debug mode toggle (hidden in an expander to not confuse regular users)
    with st.expander("Advanced Settings"):
        prev_debug = DEBUG_MODE
        DEBUG_MODE = st.checkbox(
            "Enable Debug Mode", 
            value=DEBUG_MODE,
            help="Log detailed information for troubleshooting"
        )
        if DEBUG_MODE != prev_debug:
            os.environ["VESSCO_DEBUG"] = "true" if DEBUG_MODE else "false"
            
        if DEBUG_MODE:
            st.info("Debug mode is enabled. Detailed error information and logs will be shown.")
            if st.button("Clear Debug Logs"):
                try:
                    if os.path.exists("vessco_debug.log"):
                        os.remove("vessco_debug.log")
                    if os.path.exists("llm_logs"):
                        for file in os.listdir("llm_logs"):
                            os.remove(os.path.join("llm_logs", file))
                    st.success("Debug logs cleared")
                except Exception as e:
                    st.error(f"Error clearing logs: {str(e)}")
    
    # Document Upload section
    st.header("Document Upload")
    
    # File uploader with drag and drop
    st.write("Upload documents for RAG")
    uploaded_files = st.file_uploader("Drag and drop files here", 
                                     accept_multiple_files=True,
                                     type=["pdf", "txt", "docx"],
                                     key="temp_uploader",
                                     label_visibility="collapsed")
    
    # Process files
    col1, col2 = st.columns(2)
    with col1:
        process_temp = st.button("Process Only", key="process_temp")
    with col2:
        archive_upload = st.button("Process & Archive", key="archive_upload")
    
    if uploaded_files and (process_temp or archive_upload):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a temporary directory to store the uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to the temporary directory
            for i, file in enumerate(uploaded_files):
                progress = (i) / len(uploaded_files) * 0.3
                progress_bar.progress(progress)
                status_text.text(f"Saving file {i+1}/{len(uploaded_files)}: {file.name}")
                
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
            
            # Process the documents
            status_text.text("Processing documents...")
            progress_bar.progress(0.4)
            
            # Define a callback function to update status
            def update_status(message):
                status_text.text(message)
            
            # Get LLM for document processing if using hierarchical approach
            llm = None
            if st.session_state.use_hierarchical:
                llm = get_bedrock_llm(model_id)
                # Apply debug logging in debug mode
                if DEBUG_MODE:
                    llm = monkey_patch_llm(llm)
                
            # Load and split documents with status updates
            try:
                if st.session_state.use_hierarchical and llm:
                    chunks = load_and_split_documents_hierarchically(temp_dir, llm, status_callback=update_status)
                else:
                    chunks = load_and_split_documents(temp_dir, status_callback=update_status)
            except Exception as e:
                error_details = traceback.format_exc()
                status_text.error(f"Error processing documents: {str(e)}")
                if DEBUG_MODE:
                    st.error(f"Detailed error: {error_details}")
                st.stop()
                
            progress_bar.progress(0.6)
            
            # Archive the files if requested
            if archive_upload:
                status_text.text("Archiving documents...")
                progress_bar.progress(0.7)
                
                # Create archive directory if it doesn't exist
                archive_dir = "./uploaded_documents"
                os.makedirs(archive_dir, exist_ok=True)
                
                # Copy files to archive with timestamp
                for file in uploaded_files:
                    # Add timestamp to avoid overwriting files with same name
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name_parts = os.path.splitext(file.name)
                    archive_filename = f"{file_name_parts[0]}_{timestamp}{file_name_parts[1]}"
                    
                    # Save to archive
                    archive_path = os.path.join(archive_dir, archive_filename)
                    with open(archive_path, "wb") as f:
                        f.write(file.getvalue())
            
            # Initialize or update vector database
            status_text.text("Updating vector database...")
            
            # Add detailed status updates for vector database process
            total_chunks = len(chunks)
            batch_size = max(1, total_chunks // 10)  # Create 10 update points
            
            if not st.session_state.db_initialized:
                # Initialize with detailed progress updates
                for i in range(0, total_chunks, batch_size):
                    end_idx = min(i + batch_size, total_chunks)
                    curr_batch = chunks[i:end_idx]
                    if i == 0:  # First batch initializes the database
                        initialize_vector_db(curr_batch)
                        st.session_state.db_initialized = True
                    else:  # Subsequent batches are added to the database
                        add_to_vector_db(curr_batch)
                    # Update progress
                    progress_percent = 0.8 + (0.2 * (end_idx / total_chunks))
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Updating vector database... ({end_idx}/{total_chunks} chunks)")
            else:
                # Add to existing database with progress updates
                for i in range(0, total_chunks, batch_size):
                    end_idx = min(i + batch_size, total_chunks)
                    curr_batch = chunks[i:end_idx]
                    add_to_vector_db(curr_batch)
                    # Update progress
                    progress_percent = 0.8 + (0.2 * (end_idx / total_chunks))
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Updating vector database... ({end_idx}/{total_chunks} chunks)")
            
            # Initialize or reinitialize the RAG chain
            try:
                if st.session_state.use_hierarchical:
                    st.session_state.chat_chain = create_enhanced_rag_chain(model_id=model_id)
                    # Apply debug logging in debug mode
                    if DEBUG_MODE and "llm" in st.session_state.chat_chain:
                        st.session_state.chat_chain["llm"] = monkey_patch_llm(st.session_state.chat_chain["llm"])
                else:
                    st.session_state.chat_chain = create_rag_chain(model_id=model_id)
            except Exception as e:
                error_details = traceback.format_exc()
                status_text.error(f"Error initializing chat chain: {str(e)}")
                if DEBUG_MODE:
                    st.error(f"Detailed error: {error_details}")
                st.stop()
            
            # Extract specifications from documents after RAG chain is initialized
            if st.session_state.db_initialized:
                # Initialize specification manager if needed
                if st.session_state.specification_manager is None:
                    st.session_state.specification_manager = SpecificationManager(llm)
                
                # Extract specifications from processed documents
                with st.spinner("Extracting specifications from documents..."):
                    status_text.text("Extracting specifications...")
                    specifications = st.session_state.specification_manager.extract_specifications_from_documents(
                        chunks, 
                        status_callback=update_status
                    )
                    
                    # Save specifications
                    document_id = f"upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.specification_manager.save_specifications(specifications, document_id)
                    
                    # Store in session state for display
                    st.session_state.current_specifications = specifications
                    
                    # Clear any previous product matches to force recalculation with new specs
                    st.session_state.product_matches = []
                    
                    status_text.text(f"Extracted {len(specifications)} specifications from documents")
            
            progress_bar.progress(1.0)
            if archive_upload:
                status_text.text("Processing and archiving complete!")
            else:
                status_text.text("Processing complete!")
            
            if archive_upload:
                st.success(f"Successfully processed and archived {len(uploaded_files)} documents ({len(chunks)} chunks) and extracted {len(st.session_state.current_specifications)} specifications")
            else:
                st.success(f"Successfully processed {len(chunks)} document chunks and extracted {len(st.session_state.current_specifications)} specifications")
    
    # Document Archive section
    st.header("Document Archive")
    
    # Display archived documents and allow deletion
    try:
        archived_docs = get_archive_documents()
        if archived_docs:
            st.write("Manage Archived Documents")
            
            for doc in archived_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(doc)
                with col2:
                    if st.button("Delete", key=f"del_{doc}"):
                        result = delete_document(doc)
                        if result:
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc}")
        else:
            st.info("No documents in archive. Use 'Process & Archive' to add documents.")
    except Exception as e:
        st.error(f"Error listing archived documents: {str(e)}")

    # Initialize button
    if not st.session_state.db_initialized:
        init_col1, init_col2 = st.columns(2)
        with init_col1:
            if st.button("Initialize Empty Database"):
                status = st.empty()
                status.text("Initializing empty vector database...")
                
                # Create directory if it doesn't exist
                os.makedirs("./chroma_db", exist_ok=True)
                
                # Initialize the chat chain
                try:
                    if st.session_state.use_hierarchical:
                        st.session_state.chat_chain = create_enhanced_rag_chain(model_id=model_id)
                    else:
                        st.session_state.chat_chain = create_rag_chain(model_id=model_id)
                    st.session_state.db_initialized = True
                    status.text("Initialization complete!")
                    st.success("Empty database initialized. You can now start chatting or add documents.")
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
    
    # Product Portfolio section
    st.header("Product Portfolio")
    
    # Display companies from portfolio
    portfolio_manager = st.session_state.portfolio_manager
    companies = portfolio_manager.get_companies()
    
    if companies:
        selected_company = st.selectbox("Select Company", companies)
        if selected_company:
            products = portfolio_manager.get_products_by_company(selected_company)
            st.write(f"Products for {selected_company}:")
            for product_type in products:
                for product_name, description in product_type.items():
                    if product_name != "Use Case":
                        st.write(f"- {product_name}")
                        with st.expander("Details"):
                            st.write(description)
                            st.write(f"**Use Case:** {product_type.get('Use Case', 'Not specified')}")
    else:
        st.info("No products in portfolio. Add your first product below.")
    
    # Add product form
    with st.expander("Add New Product"):
        new_company = st.text_input("Company Name")
        new_product = st.text_input("Product Name")
        new_description = st.text_area("Product Description")
        new_use_case = st.text_area("Use Case")
        
        if st.button("Add Product") and new_company and new_product and new_description:
            if portfolio_manager.add_product(new_company, new_product, new_description, new_use_case):
                st.success(f"Added {new_product} to {new_company}")
                # Refresh portfolio 
                portfolio_manager.reload_portfolio()
                st.rerun()
            else:
                st.error("Failed to add product")

# Main content area - Chat interface
# Create tabs for different views
tab1, tab2 = st.tabs(["Chat", "Analysis"])

with tab1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            if not st.session_state.db_initialized:
                response = "Please initialize the database or upload documents first."
            elif not st.session_state.chat_chain:
                response = "Chat system is not initialized. Please upload documents or initialize the database."
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Get response from RAG chain using the updated invoke method
                        response_data = st.session_state.chat_chain["invoke"]({"question": prompt})
                        
                        # Store full response data in session state for the Analysis tab
                        if "last_response_data" not in st.session_state:
                            st.session_state.last_response_data = {}
                        st.session_state.last_response_data = response_data
                        
                        # Display text response
                        if isinstance(response_data, dict) and "answer" in response_data:
                            response = response_data["answer"]
                        else:
                            response = response_data
                    except Exception as e:
                        error_details = traceback.format_exc()
                        response = f"Error generating response: {str(e)}"
                        if DEBUG_MODE:
                            response += f"\n\nDetailed error:\n```\n{error_details}\n```"
            
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.header("Document and Product Analysis")
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs(["Specifications", "Product Matches", "Document Structure"])
    
    with analysis_tabs[0]:  # Specifications tab
        st.subheader("Extracted Specifications")
        
        # Add button to scan documents for specifications
        if st.button("Scan Documents for Specifications"):
            if not st.session_state.db_initialized:
                st.error("Please initialize the database or upload documents first.")
            else:
                with st.spinner("Scanning documents for specifications..."):
                    # Get all documents from vector store
                    from embeddings import get_retriever
                    retriever = get_retriever()
                    
                    # Use a large query to get diverse documents
                    documents = retriever.get_relevant_documents(
                        "find all specifications requirements technical data"
                    )
                    
                    # Initialize specification manager if needed
                    if st.session_state.specification_manager is None:
                        st.session_state.specification_manager = SpecificationManager(
                            get_bedrock_llm(model_id)
                        )
                    
                    # Show progress
                    scan_progress = st.progress(0)
                    scan_status = st.empty()
                    
                    def update_scan_status(message):
                        scan_status.text(message)
                    
                    # Extract specifications
                    specifications = st.session_state.specification_manager.extract_specifications_from_documents(
                        documents,
                        status_callback=update_scan_status
                    )
                    
                    # Save specifications
                    document_id = f"scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.specification_manager.save_specifications(specifications, document_id)
                    
                    # Store in session state for display
                    st.session_state.current_specifications = specifications
                    
                    scan_progress.progress(1.0)
                    scan_status.text(f"Extracted {len(specifications)} specifications from documents")
                    
                    # Clear product matches to force recalculation
                    if "product_matches" in st.session_state:
                        st.session_state.product_matches = []
                    
                    # Rerun to update display
                    st.rerun()
        
        if st.session_state.current_specifications:
            # Create formatted table for display
            spec_data = []
            for spec in st.session_state.current_specifications:
                # Check if this spec has any flags
                has_flags = "flags" in spec and spec["flags"]
                flag_notes = "; ".join([f"{flag['issue_type']}" for flag in spec.get("flags", [])])
                
                # Add to table data
                spec_data.append({
                    "Item #": spec.get("item_id", ""),
                    "Label": spec.get("parameter", ""),
                    "Value": spec.get("value", ""),
                    "Unit": spec.get("unit", ""),
                    "Constraint": spec.get("constraint_type", ""),
                    "Note": spec.get("note", ""),
                    "Source": spec.get("source_document", ""),
                    "Pages": spec.get("page_number", ""),
                    "Flags": flag_notes if has_flags else ""
                })
            
            if spec_data:
                # Create DataFrame for better display
                df = pd.DataFrame(spec_data)
                
                # Apply styling to highlight flagged items
                def highlight_flags(val):
                    if val != "":
                        return 'background-color: #FFEB9C'  # Light yellow
                    return ''
                
                # Display styled dataframe
                st.dataframe(df.style.applymap(highlight_flags, subset=['Flags']))
                
                # Add download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Specifications as CSV",
                    data=csv,
                    file_name="specifications.csv",
                    mime="text/csv"
                )
        else:
            st.info("No specifications extracted yet. Upload and process documents to see specifications.")
    
    with analysis_tabs[1]:  # Product Matches tab
        st.subheader("Product Matches")
        
        if st.session_state.current_specifications:
            # Match specifications to products
            if not st.session_state.product_matches:
                with st.spinner("Matching specifications to products..."):
                    # Initialize product matcher
                    product_matcher = ProductMatcher(get_bedrock_llm(model_id))
                    
                    # Match specifications to products
                    matches = product_matcher.match_specifications(st.session_state.current_specifications)
                    st.session_state.product_matches = matches
            
            # Display product matches
            if st.session_state.product_matches:
                # Create tabs for each product
                product_tabs = st.tabs([f"{m['company']} - {m['product']} ({m['score']}%)" 
                                        for m in st.session_state.product_matches[:5]])
                
                for i, tab in enumerate(product_tabs):
                    with tab:
                        match = st.session_state.product_matches[i]
                        st.markdown(f"### {match['product']}")
                        st.markdown(f"**Company:** {match['company']}")
                        st.markdown(f"**Match Score:** {match['score']}%")
                        st.markdown("**Reasoning:**")
                        st.markdown(match['reasoning'])
                        
                        if "description" in match:
                            st.markdown("**Product Description:**")
                            st.markdown(match['description'])
                        
                        if "use_case" in match:
                            st.markdown("**Use Case:**")
                            st.markdown(match['use_case'])
            else:
                st.info("No product matches found for the extracted specifications.")
        else:
            st.info("No specifications available for matching. Upload and process documents first.")
    
    with analysis_tabs[2]:  # Document Structure tab
        st.subheader("Document Structure")
        
        # Display document structure information if available
        if "last_response_data" in st.session_state and st.session_state.last_response_data:
            data = st.session_state.last_response_data
            
            # Display document sections from last query
            if "sections_info" in data and data["sections_info"]:
                st.write("Document Sections:")
                for header, info in data["sections_info"].items():
                    st.markdown(f"- **{header}**")
                    st.markdown(f"  - Path: {info.get('header_path', 'N/A')}")
                    st.markdown(f"  - Level: {info.get('level', 'N/A')}")
            else:
                st.info("No document structure information available from last query.")
        else:
            st.info("Ask a question in the Chat tab to analyze document structure.")
    
    # Display last response data if available (keeping the existing functionality)
    if "last_response_data" in st.session_state and st.session_state.last_response_data:
        data = st.session_state.last_response_data
        
        # Create an expander for query-specific results
        with st.expander("Last Query Results"):
            # Display extracted specifications from last query
            if "specifications" in data and data["specifications"]:
                st.subheader("Query-Specific Specifications")
                
                # Create a formatted table for specifications
                spec_data = []
                for spec in data["specifications"]:
                    # Collect unique page numbers across all specifications
                    page_numbers = spec.get("page_numbers", [])
                    
                    # Create displayable source information
                    source_doc = spec.get("source_document", "Unknown")
                    page_count = len(page_numbers) if page_numbers else 0
                    page_refs = ", ".join(page_numbers) if page_numbers else "N/A"
                    
                    spec_data.append({
                        "Parameter": spec.get("parameter", ""),
                        "Value": spec.get("value", ""),
                        "Unit": spec.get("unit", ""),
                        "Constraint": spec.get("constraint_type", ""),
                        "Source": source_doc,
                        "Pages": page_count,
                        "Page Refs": page_refs
                    })
                
                if spec_data:
                    st.table(spec_data)
            else:
                st.info("No specifications extracted from the last query.")
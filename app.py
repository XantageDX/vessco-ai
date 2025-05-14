# # app.py
# import os
# import streamlit as st
# import tempfile
# import shutil
# from dotenv import load_dotenv
# import datetime

# # Import our modules
# from embeddings import load_and_split_documents, initialize_vector_db, get_retriever, add_to_vector_db
# from bedrock_llm import get_bedrock_llm
# from rag_chain import create_rag_chain
# from archive_manager import delete_document, get_archive_documents

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
            
#             # Load and split documents with status updates
#             chunks = load_and_split_documents(temp_dir, status_callback=update_status)
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
#             st.session_state.chat_chain = create_rag_chain(model_id=model_id)
            
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
#     if not st.session_state.db_initialized and st.button("Initialize Empty Database"):
#         status = st.empty()
#         status.text("Initializing empty vector database...")
        
#         # Create directory if it doesn't exist
#         os.makedirs("./chroma_db", exist_ok=True)
        
#         # Initialize the chat chain
#         try:
#             st.session_state.chat_chain = create_rag_chain(model_id=model_id)
#             st.session_state.db_initialized = True
#             status.text("Initialization complete!")
#             st.success("Empty database initialized. You can now start chatting or add documents.")
#         except Exception as e:
#             st.error(f"Initialization failed: {str(e)}")

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask me anything about your documents..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
        
#         if not st.session_state.db_initialized:
#             response = "Please initialize the database or upload documents first."
#         elif not st.session_state.chat_chain:
#             response = "Chat system is not initialized. Please upload documents or initialize the database."
#         else:
#             with st.spinner("Thinking..."):
#                 try:
#                     # Get response from RAG chain using the updated invoke method
#                     response = st.session_state.chat_chain.invoke({"question": prompt})
#                     response = response["answer"]
#                 except Exception as e:
#                     response = f"Error generating response: {str(e)}"
        
#         message_placeholder.markdown(response)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

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
#                 spec_data.append({
#                     "Parameter": spec.get("parameter", ""),
#                     "Value": spec.get("value", ""),
#                     "Unit": spec.get("unit", ""),
#                     "Constraint": spec.get("constraint_type", "")
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

#### WITH SOURCE/METADATA COLUMNS ####
# app.py
import os
import streamlit as st
import tempfile
import shutil
from dotenv import load_dotenv
import datetime
import json
import traceback

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
            
            progress_bar.progress(1.0)
            if archive_upload:
                status_text.text("Processing and archiving complete!")
            else:
                status_text.text("Processing complete!")
            
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
            
            if archive_upload:
                st.success(f"Successfully processed and archived {len(uploaded_files)} documents ({len(chunks)} chunks)")
            else:
                st.success(f"Successfully processed {len(chunks)} document chunks")
    
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
    
    # Display last response data if available
    if "last_response_data" in st.session_state and st.session_state.last_response_data:
        data = st.session_state.last_response_data
        
        # Display extracted specifications
        if "specifications" in data and data["specifications"]:
            st.subheader("Extracted Specifications")
            
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
            
        # Display product matches
        if "matches" in data and data["matches"]:
            st.subheader("Product Matches")
            
            # Create tabs for each product
            product_tabs = st.tabs([f"{m['company']} - {m['product']} ({m['score']}%)" 
                                    for m in data["matches"][:5]])
            
            for i, tab in enumerate(product_tabs):
                with tab:
                    match = data["matches"][i]
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
            st.info("No product matches found for the last query.")
    else:
        st.info("Ask a question in the Chat tab to see analysis here.")
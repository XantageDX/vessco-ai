# # embeddings.py
# import os
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_aws import BedrockEmbeddings
# from langchain_community.vectorstores import Chroma
# import chromadb
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader

# def load_and_split_documents(docs_directory):
#     """Load documents from a directory and split them into chunks."""
#     # Create loaders for different file types
#     txt_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.txt",
#         loader_cls=TextLoader
#     )
#     docx_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.docx",
#         loader_cls=Docx2txtLoader
#     )
#     # Add PDF loader
#     pdf_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.pdf",
#         loader_cls=PyPDFLoader
#     )
    
#     # Load all documents
#     txt_documents = txt_loader.load()
#     docx_documents = docx_loader.load()
#     pdf_documents = pdf_loader.load()  # Load PDF documents
#     documents = txt_documents + docx_documents + pdf_documents  # Add PDFs to the document list
#     print(f"Loaded {len(documents)} files: {len(txt_documents)} .txt, {len(docx_documents)} .docx, and {len(pdf_documents)} .pdf")
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     chunks = text_splitter.split_documents(documents)
#     return chunks

# def initialize_vector_db(chunks, persist_directory="./chroma_db"):
#     """Initialize the vector database with document chunks."""
#     # Initialize Bedrock embeddings
#     embeddings = BedrockEmbeddings(
#         model_id="amazon.titan-embed-text-v2:0",
#         region_name=os.getenv("AWS_REGION", "us-east-1"),
#     )
    
#     # Create and persist the vector store
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=persist_directory
#     )
#     return vectordb

# def get_retriever(persist_directory="./chroma_db"):
#     """Get a retriever from an existing vector database."""
#     # Initialize Bedrock embeddings
#     embeddings = BedrockEmbeddings(
#         model_id="amazon.titan-embed-text-v2:0",
#         region_name=os.getenv("AWS_REGION", "us-east-1"),
#     )
    
#     # Load the existing vector store
#     try:
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embeddings
#         )
#         return vectordb.as_retriever(search_kwargs={"k": 4})
#     except Exception as e:
#         print(f"Error loading vector database: {e}")
#         return None

# def add_to_vector_db(chunks, persist_directory="./chroma_db"):
#     """Add document chunks to an existing vector database."""
#     try:
#         # Initialize Bedrock embeddings
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v2:0",
#             region_name=os.getenv("AWS_REGION", "us-east-1"),
#         )
        
#         # Load the existing vector store
#         try:
#             vectordb = Chroma(
#                 persist_directory=persist_directory,
#                 embedding_function=embeddings
#             )
            
#             # Add the new chunks to the vector database
#             vectordb.add_documents(chunks)
            
#             # Persist the changes
#             vectordb.persist()
#             print(f"Successfully added {len(chunks)} chunks to the database")
#             return True
#         except Exception as e:
#             print(f"Error accessing vector database: {e}")
#             return False
#     except Exception as e:
#         print(f"Error adding documents to vector database: {e}")
#         return False

# def delete_document_embeddings(document_filename, persist_directory="./chroma_db"):
#     """
#     Delete all embeddings for a specific document from the vector database.
    
#     Args:
#         document_filename: Filename of the document to remove
#         persist_directory: Directory for the vector database
#     """
#     try:
#         # Initialize Bedrock embeddings
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v2:0",
#             region_name=os.getenv("AWS_REGION", "us-east-1"),
#         )
        
#         # Get the vector database
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embeddings
#         )
        
#         # Find all document chunks with matching source
#         # Note: This assumes that document metadata contains a 'source' field
#         # that matches the document filename
#         try:
#             # Query for documents with matching source in metadata
#             results = vectordb.get(
#                 where={"source": document_filename}
#             )
            
#             # If we found matching documents, delete them by ID
#             if results and results.get('ids'):
#                 vectordb.delete(ids=results['ids'])
#                 vectordb.persist()
#                 print(f"Deleted {len(results['ids'])} embeddings for {document_filename}")
#                 return True
#             else:
#                 # Try alternate approach: search each document metadata
#                 all_docs = vectordb.get()
#                 if all_docs:
#                     ids_to_delete = []
#                     for i, metadata in enumerate(all_docs.get('metadatas', [])):
#                         if metadata.get('source') == document_filename:
#                             ids_to_delete.append(all_docs['ids'][i])
                    
#                     if ids_to_delete:
#                         vectordb.delete(ids=ids_to_delete)
#                         vectordb.persist()
#                         print(f"Deleted {len(ids_to_delete)} embeddings for {document_filename}")
#                         return True
                
#                 print(f"No embeddings found for {document_filename}")
#                 return True  # Return True even if no embeddings found
#         except Exception as e:
#             print(f"Error searching for document embeddings: {e}")
#             return False
            
#     except Exception as e:
#         print(f"Error deleting document embeddings: {e}")
#         return False
# embeddings.py
# import os
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_aws import BedrockEmbeddings
# from langchain_community.vectorstores import Chroma
# import chromadb
# from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader

# def load_and_split_documents(docs_directory, status_callback=None):
#     """Load documents from a directory and split them into chunks."""
#     # Create loaders for different file types
#     txt_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.txt",
#         loader_cls=TextLoader
#     )
#     docx_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.docx",
#         loader_cls=Docx2txtLoader
#     )
#     # Add PDF loader
#     pdf_loader = DirectoryLoader(
#         docs_directory,
#         glob="**/*.pdf",
#         loader_cls=PyPDFLoader
#     )
    
#     # Update status if callback provided
#     if status_callback:
#         status_callback("Loading documents...")
    
#     # Load all documents
#     txt_documents = txt_loader.load()
#     docx_documents = docx_loader.load()
#     pdf_documents = pdf_loader.load()  # Load PDF documents
#     documents = txt_documents + docx_documents + pdf_documents  # Add PDFs to the document list
    
#     if status_callback:
#         status_callback(f"Loaded {len(documents)} files: {len(txt_documents)} .txt, {len(docx_documents)} .docx, and {len(pdf_documents)} .pdf")
    
#     # Increase chunk size for faster processing of large documents
#     if status_callback:
#         status_callback("Splitting documents into chunks...")
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,  # Increased from 500
#         chunk_overlap=100,  # Increased from 50
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     if status_callback:
#         status_callback(f"Created {len(chunks)} chunks")
    
#     return chunks

# def initialize_vector_db(chunks, persist_directory="./chroma_db"):
#     """Initialize the vector database with document chunks."""
#     # Initialize Bedrock embeddings
#     embeddings = BedrockEmbeddings(
#         model_id="amazon.titan-embed-text-v2:0",
#         region_name=os.getenv("AWS_REGION", "us-east-1"),
#     )
    
#     # Create and persist the vector store
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=persist_directory
#     )
#     return vectordb

# def get_retriever(persist_directory="./chroma_db"):
#     """Get a retriever from an existing vector database."""
#     # Initialize Bedrock embeddings
#     embeddings = BedrockEmbeddings(
#         model_id="amazon.titan-embed-text-v2:0",
#         region_name=os.getenv("AWS_REGION", "us-east-1"),
#     )
    
#     # Load the existing vector store
#     try:
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embeddings
#         )
#         return vectordb.as_retriever(search_kwargs={"k": 10})
#     except Exception as e:
#         print(f"Error loading vector database: {e}")
#         return None

# def add_to_vector_db(chunks, persist_directory="./chroma_db"):
#     """Add document chunks to an existing vector database."""
#     try:
#         # Initialize Bedrock embeddings
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v2:0",
#             region_name=os.getenv("AWS_REGION", "us-east-1"),
#         )
        
#         # Load the existing vector store
#         try:
#             vectordb = Chroma(
#                 persist_directory=persist_directory,
#                 embedding_function=embeddings
#             )
            
#             # Add the new chunks to the vector database
#             vectordb.add_documents(chunks)
            
#             # Persist the changes
#             vectordb.persist()
#             print(f"Successfully added {len(chunks)} chunks to the database")
#             return True
#         except Exception as e:
#             print(f"Error accessing vector database: {e}")
#             return False
#     except Exception as e:
#         print(f"Error adding documents to vector database: {e}")
#         return False

# def delete_document_embeddings(document_filename, persist_directory="./chroma_db"):
#     """
#     Delete all embeddings for a specific document from the vector database.
    
#     Args:
#         document_filename: Filename of the document to remove
#         persist_directory: Directory for the vector database
#     """
#     try:
#         # Initialize Bedrock embeddings
#         embeddings = BedrockEmbeddings(
#             model_id="amazon.titan-embed-text-v2:0",
#             region_name=os.getenv("AWS_REGION", "us-east-1"),
#         )
        
#         # Get the vector database
#         vectordb = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=embeddings
#         )
        
#         # Find all document chunks with matching source
#         # Note: This assumes that document metadata contains a 'source' field
#         # that matches the document filename
#         try:
#             # Query for documents with matching source in metadata
#             results = vectordb.get(
#                 where={"source": document_filename}
#             )
            
#             # If we found matching documents, delete them by ID
#             if results and results.get('ids'):
#                 vectordb.delete(ids=results['ids'])
#                 vectordb.persist()
#                 print(f"Deleted {len(results['ids'])} embeddings for {document_filename}")
#                 return True
#             else:
#                 # Try alternate approach: search each document metadata
#                 all_docs = vectordb.get()
#                 if all_docs:
#                     ids_to_delete = []
#                     for i, metadata in enumerate(all_docs.get('metadatas', [])):
#                         if metadata.get('source') == document_filename:
#                             ids_to_delete.append(all_docs['ids'][i])
                    
#                     if ids_to_delete:
#                         vectordb.delete(ids=ids_to_delete)
#                         vectordb.persist()
#                         print(f"Deleted {len(ids_to_delete)} embeddings for {document_filename}")
#                         return True
                
#                 print(f"No embeddings found for {document_filename}")
#                 return True  # Return True even if no embeddings found
#         except Exception as e:
#             print(f"Error searching for document embeddings: {e}")
#             return False
            
#     except Exception as e:
#         print(f"Error deleting document embeddings: {e}")
#         return False

# embeddings.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader

def load_and_split_documents(docs_directory, status_callback=None):
    """Load documents from a directory and split them into chunks."""
    # Create loaders for different file types
    txt_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    docx_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )
    # Add PDF loader
    pdf_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # Update status if callback provided
    if status_callback:
        status_callback("Loading documents...")
    
    # Load all documents
    txt_documents = txt_loader.load()
    docx_documents = docx_loader.load()
    pdf_documents = pdf_loader.load()  # Load PDF documents
    documents = txt_documents + docx_documents + pdf_documents  # Add PDFs to the document list
    
    if status_callback:
        status_callback(f"Loaded {len(documents)} files: {len(txt_documents)} .txt, {len(docx_documents)} .docx, and {len(pdf_documents)} .pdf")
    
    # Increase chunk size for faster processing of large documents
    if status_callback:
        status_callback("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 500
        chunk_overlap=100,  # Increased from 50
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    if status_callback:
        status_callback(f"Created {len(chunks)} chunks")
    
    return chunks

# New function for hierarchical document processing
def load_and_split_documents_hierarchically(docs_directory, llm, status_callback=None):
    """Load documents and split them with hierarchical information."""
    from document_processor import DocumentStructureExtractor
    
    # Create document structure extractor
    structure_extractor = DocumentStructureExtractor(llm)
    
    # Create loaders for different file types
    txt_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    docx_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )
    pdf_loader = DirectoryLoader(
        docs_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader  # This preserves page numbers
    )
    
    # Update status if callback provided
    if status_callback:
        status_callback("Loading documents...")
    
    # Load all documents
    txt_documents = txt_loader.load()
    docx_documents = docx_loader.load()
    pdf_documents = pdf_loader.load()
    documents = txt_documents + docx_documents + pdf_documents
    
    if status_callback:
        status_callback(f"Loaded {len(documents)} files: {len(txt_documents)} .txt, {len(docx_documents)} .docx, and {len(pdf_documents)} .pdf")
    
    # Process each document hierarchically
    all_chunks = []
    for i, document in enumerate(documents):
        if status_callback:
            status_callback(f"Analyzing structure of document {i+1}/{len(documents)}: {document.metadata.get('source', 'document')}...")
        
        # Extract document structure
        try:
            sections = structure_extractor.extract_structure(
                document.page_content,
                document.metadata.get("source", "")
            )
            
            if status_callback:
                status_callback(f"Found {len(sections)} sections in document {i+1}")
            
            # Create hierarchical chunks
            chunks = structure_extractor.create_hierarchical_chunks(sections, document.metadata)
            all_chunks.extend(chunks)
            
            if status_callback:
                status_callback(f"Processed document {i+1}/{len(documents)}, total chunks: {len(all_chunks)}")
        except Exception as e:
            print(f"Error processing document {document.metadata.get('source', '')}: {e}")
            if status_callback:
                status_callback(f"Error processing document {i+1}, skipping: {str(e)}")
    
    if status_callback:
        status_callback(f"Created {len(all_chunks)} hierarchical chunks from {len(documents)} documents")
    
    return all_chunks

def initialize_vector_db(chunks, persist_directory="./chroma_db"):
    """Initialize the vector database with document chunks."""
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    
    # Create and persist the vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def get_retriever(persist_directory="./chroma_db"):
    """Get a retriever from an existing vector database."""
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    
    # Load the existing vector store
    try:
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectordb.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

def add_to_vector_db(chunks, persist_directory="./chroma_db"):
    """Add document chunks to an existing vector database."""
    try:
        # Initialize Bedrock embeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        
        # Load the existing vector store
        try:
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            # Add the new chunks to the vector database
            vectordb.add_documents(chunks)
            
            # Persist the changes
            vectordb.persist()
            print(f"Successfully added {len(chunks)} chunks to the database")
            return True
        except Exception as e:
            print(f"Error accessing vector database: {e}")
            return False
    except Exception as e:
        print(f"Error adding documents to vector database: {e}")
        return False

def delete_document_embeddings(document_filename, persist_directory="./chroma_db"):
    """
    Delete all embeddings for a specific document from the vector database.
    
    Args:
        document_filename: Filename of the document to remove
        persist_directory: Directory for the vector database
    """
    try:
        # Initialize Bedrock embeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        
        # Get the vector database
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Find all document chunks with matching source
        # Note: This assumes that document metadata contains a 'source' field
        # that matches the document filename
        try:
            # Query for documents with matching source in metadata
            results = vectordb.get(
                where={"source": document_filename}
            )
            
            # If we found matching documents, delete them by ID
            if results and results.get('ids'):
                vectordb.delete(ids=results['ids'])
                vectordb.persist()
                print(f"Deleted {len(results['ids'])} embeddings for {document_filename}")
                return True
            else:
                # Try alternate approach: search each document metadata
                all_docs = vectordb.get()
                if all_docs:
                    ids_to_delete = []
                    for i, metadata in enumerate(all_docs.get('metadatas', [])):
                        if metadata.get('source') == document_filename:
                            ids_to_delete.append(all_docs['ids'][i])
                    
                    if ids_to_delete:
                        vectordb.delete(ids=ids_to_delete)
                        vectordb.persist()
                        print(f"Deleted {len(ids_to_delete)} embeddings for {document_filename}")
                        return True
                
                print(f"No embeddings found for {document_filename}")
                return True  # Return True even if no embeddings found
        except Exception as e:
            print(f"Error searching for document embeddings: {e}")
            return False
            
    except Exception as e:
        print(f"Error deleting document embeddings: {e}")
        return False
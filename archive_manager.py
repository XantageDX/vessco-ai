# archive_manager.py
import os
import shutil
from embeddings import load_and_split_documents, add_to_vector_db, delete_document_embeddings

def get_archive_documents():
    """
    Get a list of archived documents.
    
    Returns:
        list: List of document filenames
    """
    try:
        archive_dir = "./uploaded_documents"
        
        # Create directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        # Get list of files
        files = [f for f in os.listdir(archive_dir) 
                if os.path.isfile(os.path.join(archive_dir, f))]
        
        return sorted(files)
    except Exception as e:
        print(f"Error getting archived documents: {e}")
        return []

def delete_document(filename):
    """
    Delete a document from the archive and remove its embeddings.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        bool: Success or failure
    """
    try:
        # Set up paths
        archive_dir = "./uploaded_documents"
        file_path = os.path.join(archive_dir, filename)
        
        # Make sure the file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False
        
        # Delete the file from the archive
        os.remove(file_path)
        
        # Remove the document's embeddings
        result = delete_document_embeddings(filename)
        
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
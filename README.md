# Vessco Streamlit RAG Application - Setup Guide

## Project Structure
```
vessco-app/
├── app.py                 # Main Streamlit application file
├── embeddings.py          # Module for embeddings and vector store operations
├── bedrock_llm.py         # Module for AWS Bedrock LLM integration
├── rag_chain.py           # Module for RAG chain implementation
├── .env                   # Environment variables (AWS credentials)
├── requirements.txt       # Python dependencies
└── chroma_db/             # Directory where the vector database will be stored
```

## Setup Instructions

1. First, ensure you're in your Conda environment:
```bash
conda activate Vessco_env
```

2. Install the required packages:
```bash
pip install streamlit langchain langchain-aws langchain-community boto3 chromadb python-dotenv
pip install pypdf docx2txt
```

3. Create a new directory for your project and copy all the files:
```bash
mkdir -p vessco-app
cd vessco-app
# Copy all the files to this directory
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage Instructions

1. When the application starts, you'll see the main chat interface with a sidebar on the left.

2. In the sidebar:
   - Select the AWS Bedrock model you want to use
   - Upload documents (PDFs, TXTs, DOCXs) that you want to use as knowledge base
   - Click "Process Documents" to add them to the RAG system
   - If no documents have been added yet, you can click "Initialize Empty Database"

3. Once documents are processed, you can start chatting with the AI assistant in the main panel.
   - Type your questions in the chat input at the bottom
   - The assistant will respond with information from your documents

4. The chat history will be maintained throughout your session.

## Troubleshooting

- If you encounter errors related to AWS credentials, verify that your .env file is correctly set up
- If document processing fails, check the file formats and try with smaller files first
- If the chat doesn't respond, make sure you've initialized the database or processed documents

## Security Note

The AWS credentials in the .env file are sensitive information. In a production environment:
- Never commit .env files to source control
- Consider using AWS IAM roles instead of access keys
- Implement proper credential rotation and management
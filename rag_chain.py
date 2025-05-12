# rag_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from bedrock_llm import get_bedrock_llm
from embeddings import get_retriever

# Custom prompt for the query
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question
to be a standalone question that captures all necessary context from the conversation.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone Question:
""")

# Custom prompt for the response
QA_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user's question based on the context provided.
If the information isn't contained in the context, say that you don't know or cannot find 
the specific information, rather than making up an answer. Always be helpful, clear, and concise.

Context: {context}

Question: {question}

Answer:
""")

def create_rag_chain(persist_directory="./chroma_db", model_id=None):
    """Create a RAG chain for answering questions based on documents."""
    # Initialize retriever
    retriever = get_retriever(persist_directory=persist_directory)
    if not retriever:
        raise ValueError("Vector database could not be initialized")
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Initialize LLM
    llm = get_bedrock_llm(model_id)
    
    # Create the conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    
    return chain
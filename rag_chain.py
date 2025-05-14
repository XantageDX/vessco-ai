# # rag_chain.py
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# from bedrock_llm import get_bedrock_llm
# from embeddings import get_retriever

# # Custom prompt for the query
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
# Given the following conversation and a follow up question, rephrase the follow up question
# to be a standalone question that captures all necessary context from the conversation.

# Chat History:
# {chat_history}

# Follow Up Input: {question}

# Standalone Question:
# """)

# # Custom prompt for the response
# QA_PROMPT = PromptTemplate.from_template("""
# You are a helpful AI assistant. Answer the user's question based on the context provided.
# If the information isn't contained in the context, say that you don't know or cannot find 
# the specific information, rather than making up an answer. Always be helpful, clear, and concise.

# Context: {context}

# Question: {question}

# Answer:
# """)

# def create_rag_chain(persist_directory="./chroma_db", model_id=None):
#     """Create a RAG chain for answering questions based on documents."""
#     # Initialize retriever
#     retriever = get_retriever(persist_directory=persist_directory)
#     if not retriever:
#         raise ValueError("Vector database could not be initialized")
    
#     # Initialize memory
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
    
#     # Initialize LLM
#     llm = get_bedrock_llm(model_id)
    
#     # Create the conversational chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT}
#     )
    
#     return chain

# # rag_chain.py
# import json
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# from bedrock_llm import get_bedrock_llm
# from embeddings import get_retriever
# from spec_extractor import SpecificationExtractor
# from product_matcher import ProductMatcher

# # Custom prompt for the query
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
# Given the following conversation and a follow up question, rephrase the follow up question
# to be a standalone question that captures all necessary context from the conversation.

# Chat History:
# {chat_history}

# Follow Up Input: {question}

# Standalone Question:
# """)

# # Custom prompt for the response
# QA_PROMPT = PromptTemplate.from_template("""
# You are a helpful AI assistant. Answer the user's question based on the context provided.
# If the information isn't contained in the context, say that you don't know or cannot find 
# the specific information, rather than making up an answer. Always be helpful, clear, and concise.

# Context: {context}

# Question: {question}

# Answer:
# """)

# def create_rag_chain(persist_directory="./chroma_db", model_id=None):
#     """Create a RAG chain for answering questions based on documents."""
#     # Initialize retriever
#     retriever = get_retriever(persist_directory=persist_directory)
#     if not retriever:
#         raise ValueError("Vector database could not be initialized")
    
#     # Initialize memory
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
    
#     # Initialize LLM
#     llm = get_bedrock_llm(model_id)
    
#     # Create the conversational chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#         combine_docs_chain_kwargs={"prompt": QA_PROMPT}
#     )
    
#     return chain

# def create_enhanced_rag_chain(persist_directory="./chroma_db", model_id=None):
#     """Create an enhanced RAG chain with two-level querying and product matching."""
#     # Initialize components
#     retriever = get_retriever(persist_directory)
#     if not retriever:
#         raise ValueError("Vector database could not be initialized")
        
#     llm = get_bedrock_llm(model_id)
#     spec_extractor = SpecificationExtractor(llm)
#     product_matcher = ProductMatcher(llm)
    
#     # Initialize memory
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
    
#     # Custom chain function that processes queries
#     def process_query(input_dict):
#         try:
#             question = input_dict.get("question", "")
            
#             # Get chat history from memory
#             chat_history = memory.load_memory_variables({})
#             history = chat_history.get("chat_history", [])
            
#             # Step 1: Generate standalone question from the conversation context
#             if history:
#                 try:
#                     # Use the LLM to generate a standalone question
#                     condense_prompt = CONDENSE_QUESTION_PROMPT.format(
#                         chat_history=history,
#                         question=question
#                     )
#                     standalone_question = llm.invoke(condense_prompt).content.strip()
#                 except Exception as e:
#                     print(f"Error generating standalone question: {e}")
#                     standalone_question = question
#             else:
#                 standalone_question = question
            
#             # Step 2: First level RAG to find relevant sections
#             try:
#                 initial_results = retriever.get_relevant_documents(standalone_question)
#             except Exception as e:
#                 print(f"Error retrieving documents: {e}")
#                 initial_results = []
            
#             # Extract header information from results
#             sections_info = {}
#             for doc in initial_results:
#                 header = doc.metadata.get("header", "")
#                 if header and header not in sections_info:
#                     sections_info[header] = {
#                         "header_path": doc.metadata.get("header_path", ""),
#                         "level": doc.metadata.get("level", 0)
#                     }
            
#             # Step 3: Extract specifications from relevant sections
#             all_specs = []
#             for doc in initial_results:
#                 try:
#                     specs = spec_extractor.extract_specifications(doc.page_content)
#                     all_specs.extend(specs)
#                 except Exception as e:
#                     print(f"Error extracting specifications: {e}")
#                     # Continue with other documents
#                     continue
            
#             # Deduplicate specifications
#             unique_specs = []
#             seen_params = set()
#             for spec in all_specs:
#                 try:
#                     param_key = f"{spec['parameter']}:{spec['value']}"
#                     if param_key not in seen_params:
#                         seen_params.add(param_key)
#                         unique_specs.append(spec)
#                 except Exception as e:
#                     print(f"Error deduplicating specification: {e}")
#                     # Add it anyway if we can't deduplicate
#                     unique_specs.append(spec)
            
#             # Step 4: Match specifications to products
#             try:
#                 product_matches = product_matcher.match_specifications(unique_specs)
#             except Exception as e:
#                 print(f"Error matching products: {e}")
#                 product_matches = []
            
#             # Step 5: Generate the final response
#             sections_text = "\n".join([
#                 f"- {info['header_path']}" for header, info in sections_info.items()
#             ]) if sections_info else "No specific document sections were found relevant to this query."
            
#             specs_text = ""
#             if unique_specs:
#                 specs_text = json.dumps(unique_specs, indent=2)
#             else:
#                 specs_text = "No specific technical specifications were identified in the documents."
            
#             matches_text = ""
#             if product_matches:
#                 for match in product_matches[:5]:  # Top 5 matches
#                     matches_text += f"- {match['company']} - {match['product']}: {match['score']}% match\n"
#                     matches_text += f"  Reasoning: {match['reasoning']}\n\n"
#             else:
#                 matches_text = "No product matches were found for the specifications."
            
#             # Generate comprehensive response
#             try:
#                 prompt = f"""
#                 Based on the user's query: "{standalone_question}", I found these relevant document sections:
                
#                 {sections_text}
                
#                 From these sections, I extracted these specifications:
                
#                 {specs_text}
                
#                 These specifications match the following products in our portfolio:
                
#                 {matches_text}
                
#                 Provide a comprehensive answer to the user's query that:
#                 1. Addresses their specific question directly and succinctly
#                 2. Summarizes the key specifications found in the documents
#                 3. Recommends the top matching products with brief explanations of why they're a good fit
#                 4. Is well-organized with clear sections
#                 """
                
#                 final_response = llm.invoke(prompt).content
#             except Exception as e:
#                 print(f"Error generating final response: {e}")
#                 # Fallback response if the final generation fails
#                 final_response = f"""
#                 I found some information related to your query about "{standalone_question}".
                
#                 I identified {len(unique_specs)} specifications in the relevant document sections.
                
#                 Based on these specifications, the top matching products are:
#                 {matches_text if matches_text else "No strong product matches were found."}
                
#                 Please note that I encountered an issue generating a more detailed response.
#                 """
            
#             # Add the exchange to memory
#             try:
#                 memory.save_context(
#                     {"input": question},
#                     {"output": final_response}
#                 )
#             except Exception as e:
#                 print(f"Error saving to memory: {e}")
            
#             return {
#                 "answer": final_response,
#                 "specifications": unique_specs,
#                 "matches": product_matches
#             }
#         except Exception as e:
#             error_message = f"Error processing query: {e}"
#             print(error_message)
#             return {
#                 "answer": f"I'm sorry, I encountered an error while processing your question. Please try asking a different question or rephrasing your query. Error details: {str(e)}",
#                 "specifications": [],
#                 "matches": []
#             }
    
#     # Return the chain object
#     return {
#         "invoke": process_query,
#         "memory": memory,
#         "llm": llm,
#         "retriever": retriever
#     }

#### WITH SOURCE/METADATA COLUMNS ####
# rag_chain.py
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from bedrock_llm import get_bedrock_llm
from embeddings import get_retriever
from spec_extractor import SpecificationExtractor
from product_matcher import ProductMatcher

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

def create_enhanced_rag_chain(persist_directory="./chroma_db", model_id=None):
    """Create an enhanced RAG chain with two-level querying and product matching."""
    # Initialize components
    retriever = get_retriever(persist_directory)
    if not retriever:
        raise ValueError("Vector database could not be initialized")
        
    llm = get_bedrock_llm(model_id)
    spec_extractor = SpecificationExtractor(llm)
    product_matcher = ProductMatcher(llm)
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Custom chain function that processes queries
    def process_query(input_dict):
        try:
            question = input_dict.get("question", "")
            
            # Get chat history from memory
            chat_history = memory.load_memory_variables({})
            history = chat_history.get("chat_history", [])
            
            # Step 1: Generate standalone question from the conversation context
            if history:
                try:
                    # Use the LLM to generate a standalone question
                    condense_prompt = CONDENSE_QUESTION_PROMPT.format(
                        chat_history=history,
                        question=question
                    )
                    standalone_question = llm.invoke(condense_prompt).content.strip()
                except Exception as e:
                    print(f"Error generating standalone question: {e}")
                    standalone_question = question
            else:
                standalone_question = question
            
            # Step 2: First level RAG to find relevant sections
            try:
                initial_results = retriever.get_relevant_documents(standalone_question)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                initial_results = []
            
            # Extract header information from results
            sections_info = {}
            for doc in initial_results:
                header = doc.metadata.get("header", "")
                if header and header not in sections_info:
                    sections_info[header] = {
                        "header_path": doc.metadata.get("header_path", ""),
                        "level": doc.metadata.get("level", 0)
                    }
            
            # Step 3: Extract specifications from relevant sections
            all_specs = []
            for doc in initial_results:
                try:
                    specs = spec_extractor.extract_specifications(doc.page_content, doc.metadata)
                    
                    # Collect page numbers across all specs for aggregate counting
                    doc_pages = set()
                    if "page" in doc.metadata:
                        doc_pages.add(str(doc.metadata["page"]))
                    elif "page_number" in doc.metadata:
                        doc_pages.add(str(doc.metadata["page_number"]))
                    
                    all_specs.extend(specs)
                except Exception as e:
                    print(f"Error extracting specifications: {e}")
                    # Continue with other documents
                    continue
            
            # Deduplicate specifications
            unique_specs = []
            seen_params = {}  # Using dict to track both seen params and their indices
            
            for spec_idx, spec in enumerate(all_specs):
                try:
                    param_key = f"{spec['parameter']}:{spec['value']}"
                    
                    if param_key in seen_params:
                        # Get the existing spec index
                        existing_idx = seen_params[param_key]
                        existing_spec = unique_specs[existing_idx]
                        
                        # Merge page numbers
                        if "page_numbers" in spec and "page_numbers" in existing_spec:
                            combined_pages = list(set(existing_spec["page_numbers"] + spec["page_numbers"]))
                            unique_specs[existing_idx]["page_numbers"] = combined_pages
                        
                        # Update source document if it's different
                        if "source_document" in spec and spec["source_document"] != existing_spec.get("source_document"):
                            # If multiple sources, note that in the source field
                            if "," not in existing_spec.get("source_document", ""):
                                unique_specs[existing_idx]["source_document"] = f"Multiple sources"
                    else:
                        seen_params[param_key] = len(unique_specs)
                        unique_specs.append(spec)
                except Exception as e:
                    print(f"Error deduplicating specification: {e}")
                    # Add it anyway if we can't deduplicate
                    unique_specs.append(spec)
            
            # Step 4: Match specifications to products
            try:
                product_matches = product_matcher.match_specifications(unique_specs)
            except Exception as e:
                print(f"Error matching products: {e}")
                product_matches = []
            
            # Step 5: Generate the final response
            sections_text = "\n".join([
                f"- {info['header_path']}" for header, info in sections_info.items()
            ]) if sections_info else "No specific document sections were found relevant to this query."
            
            specs_text = ""
            if unique_specs:
                specs_text = json.dumps(unique_specs, indent=2)
            else:
                specs_text = "No specific technical specifications were identified in the documents."
            
            matches_text = ""
            if product_matches:
                for match in product_matches[:5]:  # Top 5 matches
                    matches_text += f"- {match['company']} - {match['product']}: {match['score']}% match\n"
                    matches_text += f"  Reasoning: {match['reasoning']}\n\n"
            else:
                matches_text = "No product matches were found for the specifications."
            
            # Generate comprehensive response
            try:
                prompt = f"""
                Based on the user's query: "{standalone_question}", I found these relevant document sections:
                
                {sections_text}
                
                From these sections, I extracted these specifications:
                
                {specs_text}
                
                These specifications match the following products in our portfolio:
                
                {matches_text}
                
                Provide a comprehensive answer to the user's query that:
                1. Addresses their specific question directly and succinctly
                2. Summarizes the key specifications found in the documents
                3. Recommends the top matching products with brief explanations of why they're a good fit
                4. Is well-organized with clear sections
                """
                
                final_response = llm.invoke(prompt).content
            except Exception as e:
                print(f"Error generating final response: {e}")
                # Fallback response if the final generation fails
                final_response = f"""
                I found some information related to your query about "{standalone_question}".
                
                I identified {len(unique_specs)} specifications in the relevant document sections.
                
                Based on these specifications, the top matching products are:
                {matches_text if matches_text else "No strong product matches were found."}
                
                Please note that I encountered an issue generating a more detailed response.
                """
            
            # Add the exchange to memory
            try:
                memory.save_context(
                    {"input": question},
                    {"output": final_response}
                )
            except Exception as e:
                print(f"Error saving to memory: {e}")
            
            return {
                "answer": final_response,
                "specifications": unique_specs,
                "matches": product_matches
            }
        except Exception as e:
            error_message = f"Error processing query: {e}"
            print(error_message)
            return {
                "answer": f"I'm sorry, I encountered an error while processing your question. Please try asking a different question or rephrasing your query. Error details: {str(e)}",
                "specifications": [],
                "matches": []
            }
    
    # Return the chain object
    return {
        "invoke": process_query,
        "memory": memory,
        "llm": llm,
        "retriever": retriever
    }
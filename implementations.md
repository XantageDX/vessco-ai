# Vessco AI - Technical Implementation Plan

## Overview

We'll enhance the existing Vessco RAG application with hierarchical document processing, structured specification extraction, and product matching capabilities. This implementation plan outlines the changes we'll make to the codebase.

## New File Structure

```
vessco-app/
├── app.py                      # Updated main Streamlit application
├── embeddings.py               # Enhanced with hierarchical chunking
├── bedrock_llm.py              # Remains largely unchanged
├── rag_chain.py                # Enhanced with two-level querying
├── document_processor.py       # NEW: Handles document structure extraction
├── spec_extractor.py           # NEW: Extracts specifications from text
├── product_matcher.py          # NEW: Matches specs to product portfolio
├── portfolio_manager.py        # NEW: Loads and manages product data
├── .env                        # Environment variables
├── requirements.txt            # Updated Python dependencies
└── chroma_db/                  # Vector database storage
```

## Implementation Details

### 1. Document Processor Module

This new module will handle hierarchical document structure extraction:

```python
# document_processor.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentStructureExtractor:
    """Extract hierarchical structure from documents."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_structure(self, text, filename):
        """
        Extract document structure using regex and LLM assistance.
        Returns a list of sections with hierarchy information.
        """
        # First pass: identify potential section headers using regex
        header_patterns = [
            r'(?m)^(SECTION|DIVISION)\s+(\d+[.\d]*)\s*[-–]\s*(.+)$',  # CSI format
            r'(?m)^(\d+[.\d]*)\s+(.+)$',  # Numbered sections
            r'(?m)^([A-Z][A-Z\s]+)$'  # ALL CAPS headers
        ]
        
        potential_headers = []
        for pattern in header_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                potential_headers.append((match.start(), match.group()))
        
        # Sort by position in document
        potential_headers.sort()
        
        # Second pass: use LLM to validate headers and determine hierarchy
        sections = []
        for i, (pos, header_text) in enumerate(potential_headers):
            next_pos = potential_headers[i+1][0] if i < len(potential_headers)-1 else len(text)
            section_text = text[pos:next_pos]
            
            # Use LLM to validate this is a real section header
            # and determine its level in the hierarchy
            prompt = f"""
            Analyze this potential document section header and content:
            
            HEADER: {header_text}
            
            CONTENT PREVIEW: {section_text[:500]}...
            
            Is this a valid section header in a technical specification document?
            If yes, what level in the hierarchy is it (1=highest, 5=lowest)?
            Respond with only: VALID,LEVEL or INVALID
            """
            
            response = self.llm.invoke(prompt).content.strip()
            
            if response.startswith("VALID"):
                _, level = response.split(",")
                sections.append({
                    "text": section_text,
                    "header": header_text,
                    "level": int(level),
                    "position": pos
                })
        
        return sections
    
    def create_hierarchical_chunks(self, sections, metadata=None):
        """
        Create document chunks with hierarchical metadata.
        """
        base_metadata = metadata or {}
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        hierarchical_chunks = []
        
        for section in sections:
            # Create path string (breadcrumb) of headers
            header_path = self._create_header_path(section, sections)
            
            # Create section metadata
            section_metadata = {
                **base_metadata,
                "header": section["header"],
                "level": section["level"],
                "header_path": header_path
            }
            
            # Split section into chunks
            chunks = text_splitter.create_documents(
                [section["text"]], 
                metadatas=[section_metadata]
            )
            
            hierarchical_chunks.extend(chunks)
        
        return hierarchical_chunks
    
    def _create_header_path(self, section, all_sections):
        """Create hierarchical path for a section based on its level."""
        # Find parent sections (sections with lower level numbers that appear before this one)
        level = section["level"]
        position = section["position"]
        
        path_parts = []
        for level_search in range(1, level):
            # Find the latest section with this level that appears before current section
            parent_candidates = [
                s for s in all_sections 
                if s["level"] == level_search and s["position"] < position
            ]
            
            if parent_candidates:
                # Get the latest parent at this level
                latest_parent = max(parent_candidates, key=lambda x: x["position"])
                path_parts.append(latest_parent["header"])
        
        # Add current section header
        path_parts.append(section["header"])
        
        return " > ".join(path_parts)
```

### 2. Specification Extractor Module

This module will extract structured specifications from text:

```python
# spec_extractor.py
from typing import List, Dict, Any

class SpecificationExtractor:
    """Extract structured specifications from technical documents."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_specifications(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key specifications from text.
        Returns a list of structured specification objects.
        """
        prompt = f"""
        Extract technical specifications from the following text. Focus on:
        - Dimensions (size, capacity, flow rates)
        - Materials
        - Performance requirements
        - Compliance standards
        - Operating conditions
        
        Format each specification as a JSON object with these fields:
        - parameter: The parameter being specified
        - value: The required value or range
        - unit: The unit of measurement (if applicable)
        - constraint_type: "minimum", "maximum", "exact", or "range"
        
        TEXT:
        {text}
        
        SPECIFICATIONS (JSON array):
        """
        
        response = self.llm.invoke(prompt).content
        
        # Parse the response to get specifications
        # In practice, you'd need more robust parsing and error handling
        import json
        try:
            # Find JSON in the response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                specifications = json.loads(json_str)
                return specifications
            return []
        except:
            # Fallback if parsing fails
            return []
```

### 3. Product Matcher Module

This module will match extracted specifications to your product portfolio:

```python
# product_matcher.py
import json
from typing import List, Dict, Any
import re

class ProductMatcher:
    """Match extracted specifications to product portfolio."""
    
    def __init__(self, llm, portfolio_path="portfolio.json"):
        self.llm = llm
        with open(portfolio_path, 'r') as f:
            self.portfolio = json.load(f)
    
    def match_specifications(self, specifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match specifications to products in the portfolio.
        Returns a list of matching products with confidence scores.
        """
        if not specifications:
            return []
        
        # Convert specifications to a string representation
        spec_text = json.dumps(specifications, indent=2)
        
        # For each product in portfolio, evaluate match
        matches = []
        
        for product_entry in self.portfolio.get("portfolio", []):
            for company, company_data in product_entry.items():
                for product_type in company_data.get("Tipes", []):
                    for product_name, product_description in product_type.items():
                        if product_name == "Use Case":
                            continue
                        
                        # Combine description and use case
                        use_case = product_type.get("Use Case", "")
                        full_description = f"{product_description}\n\nUse Case: {use_case}"
                        
                        # Have the LLM evaluate the match
                        prompt = f"""
                        Evaluate how well the following product matches the specifications:
                        
                        SPECIFICATIONS:
                        {spec_text}
                        
                        PRODUCT: {product_name}
                        COMPANY: {company}
                        DESCRIPTION: {full_description}
                        
                        Provide a match score from 0-100 where:
                        - 0: No match at all
                        - 50: Partially matches some requirements
                        - 100: Perfect match for all specifications
                        
                        Then explain your reasoning briefly.
                        
                        FORMAT: Score: [0-100]\nReasoning: [explanation]
                        """
                        
                        response = self.llm.invoke(prompt).content.strip()
                        
                        # Parse the response
                        score_match = re.search(r'Score:\s*(\d+)', response)
                        reasoning_match = re.search(r'Reasoning:\s*(.*)', response, re.DOTALL)
                        
                        if score_match:
                            score = int(score_match.group(1))
                            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                            
                            matches.append({
                                "company": company,
                                "product": product_name,
                                "score": score,
                                "reasoning": reasoning
                            })
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches
```

### 4. Portfolio Manager Module

This module will handle loading and accessing the product portfolio:

```python
# portfolio_manager.py
import json
import os
from typing import Dict, List, Any

class PortfolioManager:
    """Manage product portfolio data."""
    
    def __init__(self, portfolio_path="portfolio.json"):
        """Initialize with path to portfolio data."""
        self.portfolio_path = portfolio_path
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """Load portfolio from JSON file."""
        if not os.path.exists(self.portfolio_path):
            return {"portfolio": []}
        
        with open(self.portfolio_path, 'r') as f:
            return json.load(f)
    
    def get_companies(self) -> List[str]:
        """Get list of companies in portfolio."""
        companies = []
        for product_entry in self.portfolio.get("portfolio", []):
            companies.extend(product_entry.keys())
        return companies
    
    def get_products_by_company(self, company: str) -> List[Dict[str, Any]]:
        """Get products for a specific company."""
        for product_entry in self.portfolio.get("portfolio", []):
            if company in product_entry:
                return product_entry[company].get("Tipes", [])
        return []
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products with company information."""
        all_products = []
        for product_entry in self.portfolio.get("portfolio", []):
            for company, company_data in product_entry.items():
                for product_type in company_data.get("Tipes", []):
                    for product_name, product_description in product_type.items():
                        if product_name != "Use Case":
                            all_products.append({
                                "company": company,
                                "product": product_name,
                                "description": product_description,
                                "use_case": product_type.get("Use Case", "")
                            })
        return all_products
```

### 5. Updated Embeddings Module

Enhance the existing embeddings.py file to support hierarchical chunking:

```python
# Updates to embeddings.py

def load_and_split_documents_hierarchically(docs_directory, llm, status_callback=None):
    """Load documents and split them with hierarchical information."""
    from document_processor import DocumentStructureExtractor
    
    # Create document structure extractor
    structure_extractor = DocumentStructureExtractor(llm)
    
    # [Existing loading code from your current function]
    # ...
    
    # Instead of directly splitting, process each document hierarchically
    all_chunks = []
    for document in documents:
        if status_callback:
            status_callback(f"Analyzing structure of {document.metadata.get('source', 'document')}...")
        
        # Extract document structure
        sections = structure_extractor.extract_structure(
            document.page_content,
            document.metadata.get("source", "")
        )
        
        # Create hierarchical chunks
        chunks = structure_extractor.create_hierarchical_chunks(sections, document.metadata)
        all_chunks.extend(chunks)
    
    if status_callback:
        status_callback(f"Created {len(all_chunks)} hierarchical chunks")
    
    return all_chunks

# Add this function to your existing embeddings.py
```

### 6. Enhanced RAG Chain

Update the RAG chain to support the two-level query process:

```python
# Updates to rag_chain.py

def create_enhanced_rag_chain(persist_directory="./chroma_db", model_id=None):
    """Create an enhanced RAG chain with two-level querying and product matching."""
    from spec_extractor import SpecificationExtractor
    from product_matcher import ProductMatcher
    
    # Initialize components
    retriever = get_retriever(persist_directory)
    llm = get_bedrock_llm(model_id)
    spec_extractor = SpecificationExtractor(llm)
    product_matcher = ProductMatcher(llm)
    
    # Custom chain function
    def process_query(input_query):
        # Step 1: First level RAG to find relevant sections
        initial_results = retriever.get_relevant_documents(input_query)
        
        # Extract header information from results
        sections_info = {}
        for doc in initial_results:
            header = doc.metadata.get("header", "")
            if header and header not in sections_info:
                sections_info[header] = {
                    "header_path": doc.metadata.get("header_path", ""),
                    "level": doc.metadata.get("level", 0)
                }
        
        # Step 2: Extract specifications from relevant sections
        all_specs = []
        for doc in initial_results:
            specs = spec_extractor.extract_specifications(doc.page_content)
            all_specs.extend(specs)
        
        # Deduplicate specifications
        unique_specs = []
        seen_params = set()
        for spec in all_specs:
            param_key = f"{spec['parameter']}:{spec['value']}"
            if param_key not in seen_params:
                seen_params.add(param_key)
                unique_specs.append(spec)
        
        # Step 3: Match specifications to products
        product_matches = product_matcher.match_specifications(unique_specs)
        
        # Step 4: Generate the final response
        sections_text = "\n".join([
            f"- {info['header_path']}" for header, info in sections_info.items()
        ])
        
        specs_text = json.dumps(unique_specs, indent=2)
        
        matches_text = ""
        for match in product_matches[:5]:  # Top 5 matches
            matches_text += f"- {match['company']} - {match['product']}: {match['score']}% match\n"
            matches_text += f"  Reasoning: {match['reasoning']}\n\n"
        
        # Generate comprehensive response
        prompt = f"""
        Based on the user's query: "{input_query}", I found these relevant document sections:
        
        {sections_text}
        
        From these sections, I extracted these specifications:
        
        {specs_text}
        
        These specifications match the following products in our portfolio:
        
        {matches_text}
        
        Provide a comprehensive answer to the user's query that:
        1. Addresses their specific question
        2. Summarizes the key specifications found
        3. Recommends the top matching products with brief explanations
        4. Is well-organized with clear sections
        """
        
        final_response = llm.invoke(prompt).content
        
        return {
            "answer": final_response,
            "specifications": unique_specs,
            "matches": product_matches
        }
    
    # Return a simple chain function
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    return {
        "invoke": process_query
    }

# Add this function to your existing rag_chain.py
```

### 7. Updated App.py

Finally, modify the Streamlit app to use these new capabilities:

```python
# Updates to app.py

# Add these imports
from document_processor import DocumentStructureExtractor
from spec_extractor import SpecificationExtractor
from product_matcher import ProductMatcher
from portfolio_manager import PortfolioManager

# Replace create_rag_chain with create_enhanced_rag_chain in relevant places
from rag_chain import create_enhanced_rag_chain

# Add portfolio visualization in sidebar
with st.sidebar:
    # ... existing sidebar code ...
    
    # Add portfolio section
    st.header("Product Portfolio")
    portfolio_manager = PortfolioManager()
    companies = portfolio_manager.get_companies()
    
    selected_company = st.selectbox("Select Company", companies)
    if selected_company:
        products = portfolio_manager.get_products_by_company(selected_company)
        st.write(f"Products for {selected_company}:")
        for product_type in products:
            for product_name in product_type.keys():
                if product_name != "Use Case":
                    st.write(f"- {product_name}")

# Update document processing to use hierarchical chunking
if uploaded_files and (process_temp or archive_upload):
    # ... existing code ...
    
    # Get LLM for document processing
    llm = get_bedrock_llm(model_id)
    
    # Use hierarchical chunking instead
    chunks = load_and_split_documents_hierarchically(temp_dir, llm, status_callback=update_status)
    
    # ... rest of existing code ...

# Update the chat response display to include specifications and matches
if response and isinstance(response, dict) and "specifications" in response:
    # Display answer
    message_placeholder.markdown(response["answer"])
    
    # Add expanders for specifications and matches
    with st.expander("View Extracted Specifications"):
        st.json(response["specifications"])
    
    with st.expander("View Product Matches"):
        for match in response["matches"][:5]:
            st.markdown(f"**{match['company']} - {match['product']}** ({match['score']}% match)")
            st.markdown(f"_{match['reasoning']}_")
            st.markdown("---")
```

## Required Updates to requirements.txt

```
streamlit>=1.28.0
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-aws>=0.1.0
langchain-community>=0.0.16
boto3>=1.29.0
chromadb>=0.4.22
pypdf>=4.0.0
docx2txt>=0.8
regex>=2022.10.31
```

## Implementation Approach

1. Create the new modules first (document_processor.py, spec_extractor.py, etc.)
2. Update embeddings.py to support hierarchical chunking
3. Enhance rag_chain.py with the two-level query process
4. Finally update app.py to use these new capabilities
5. Test with sample construction documents

This implementation preserves your existing functionality while adding the new capabilities we discussed.
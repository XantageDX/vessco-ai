# Vessco AI: Automatic Specification Table Generation

## Overview

This document outlines the integration plan for adding automatic specification table generation to the Vessco AI application. The goal is to extract detailed specifications from uploaded documents and present them in a structured table with the following columns:

```
Item # | Label | Value | Unit | Constraint | Note | Source | Pages
```

The extracted specifications will be available immediately after document processing, rather than only when responding to queries.

## Implementation Steps

### 1. Fix Page Number Extraction

**Issue**: Current implementation doesn't correctly capture and display document page numbers.

**Solution**:

```python
# In embeddings.py - Enhance the PyPDFLoader handling
def load_and_split_documents_hierarchically(docs_directory, llm, status_callback=None):
    # ...existing code...
    
    # For PDF documents, ensure page numbers are properly preserved
    for i, document in enumerate(documents):
        if document.metadata.get("source", "").lower().endswith(".pdf"):
            # Ensure page numbers start from 1 (human-readable) not 0 (zero-indexed)
            if "page" in document.metadata:
                document.metadata["page_number"] = document.metadata["page"] + 1
            elif "page_number" in document.metadata:
                document.metadata["page_number"] = int(document.metadata["page_number"]) + 1
    
    # ...rest of existing code...
```

### 2. Create Specification Storage Manager

**New File**: `specification_manager.py`

```python
# specification_manager.py
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

class SpecificationManager:
    """Manage extraction and storage of specifications from documents."""
    
    def __init__(self, llm, storage_dir="./extracted_specs"):
        """Initialize the specification manager."""
        self.llm = llm
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def extract_specifications_from_documents(self, documents, status_callback=None):
        """
        Extract specifications from a list of documents.
        
        Args:
            documents: List of document objects with text and metadata
            status_callback: Optional callback for progress updates
            
        Returns:
            List of extracted specifications
        """
        all_specifications = []
        
        # Group documents by source to process one source file at a time
        documents_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in documents_by_source:
                documents_by_source[source] = []
            documents_by_source[source].append(doc)
        
        total_sources = len(documents_by_source)
        for i, (source, source_docs) in enumerate(documents_by_source.items()):
            if status_callback:
                status_callback(f"Extracting specifications from {source} ({i+1}/{total_sources})")
            
            # Step 1: First pass - identify sections likely to contain specifications
            specification_sections = self._identify_specification_sections(source_docs)
            
            if status_callback:
                status_callback(f"Found {len(specification_sections)} potential specification sections in {source}")
            
            # Step 2: Extract specifications from identified sections
            source_specifications = []
            for j, section in enumerate(specification_sections):
                if status_callback:
                    status_callback(f"Processing section {j+1}/{len(specification_sections)}")
                
                # Extract specifications from this section
                section_specs = self._extract_specifications_from_section(
                    section["text"], 
                    section["metadata"],
                    f"Item_{i+1:03d}"  # Base item_id for this source
                )
                
                source_specifications.extend(section_specs)
            
            # Step 3: Validate and flag potential issues
            validated_specs = self._validate_specifications(source_specifications)
            
            # Add unique IDs if needed
            for spec in validated_specs:
                if "id" not in spec:
                    spec["id"] = str(uuid.uuid4())
            
            all_specifications.extend(validated_specs)
            
            if status_callback:
                status_callback(f"Extracted {len(validated_specs)} specifications from {source}")
        
        return all_specifications
    
    def _identify_specification_sections(self, documents):
        """
        Identify sections in documents likely to contain specifications.
        
        Returns:
            List of sections with text and metadata
        """
        specification_sections = []
        
        # Combine sequential chunks from the same source and page
        current_section = None
        
        for doc in sorted(documents, key=lambda d: (d.metadata.get("source", ""), 
                                                   d.metadata.get("page_number", 0))):
            # Check if this is a new section
            if current_section is None or \
               current_section["metadata"]["source"] != doc.metadata.get("source", "") or \
               current_section["metadata"]["page_number"] != doc.metadata.get("page_number", 0):
                
                # Save the previous section if it exists
                if current_section is not None:
                    # Check if section likely contains specifications
                    if self._section_contains_specifications(current_section["text"]):
                        specification_sections.append(current_section)
                
                # Start a new section
                current_section = {
                    "text": doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get("source", ""),
                        "page_number": doc.metadata.get("page_number", 0),
                        "header": doc.metadata.get("header", ""),
                        "header_path": doc.metadata.get("header_path", "")
                    }
                }
            else:
                # Append to current section
                current_section["text"] += "\n" + doc.page_content
        
        # Check the last section
        if current_section is not None and self._section_contains_specifications(current_section["text"]):
            specification_sections.append(current_section)
        
        return specification_sections
    
    def _section_contains_specifications(self, text):
        """
        Check if a section likely contains specifications.
        
        Returns:
            Boolean indicating if section likely contains specifications
        """
        # Simple heuristic check for now
        keywords = [
            "specification", "requirement", "shall", "must", "minimum", "maximum",
            "not less than", "not more than", "equal to", "rating", "capacity"
        ]
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        
        # Also check for patterns that might indicate specifications
        patterns = [
            r"\d+\s*[a-zA-Z]+",  # Number followed by unit
            r"[<>]=?\s*\d+",     # Inequality with number
            r"\d+\s*[-–]\s*\d+"  # Number range
        ]
        
        import re
        pattern_count = sum(1 for pattern in patterns if re.search(pattern, text))
        
        # Return true if we have enough evidence of specifications
        return keyword_count >= 2 or pattern_count >= 2
    
    def _extract_specifications_from_section(self, text, metadata, base_item_id):
        """
        Extract specifications from a section of text.
        
        Args:
            text: The text to extract specifications from
            metadata: Document metadata
            base_item_id: Base item ID to use for these specifications
            
        Returns:
            List of extracted specifications
        """
        prompt = self._build_extraction_prompt(text, base_item_id)
        response = self.llm.invoke(prompt).content
        
        # Parse the response to extract JSON objects
        specifications = self._parse_specifications_from_response(response)
        
        # Add metadata to specifications
        for spec in specifications:
            spec["source_document"] = metadata.get("source", "Unknown")
            spec["page_number"] = metadata.get("page_number", 0)
            spec["header"] = metadata.get("header", "")
            spec["header_path"] = metadata.get("header_path", "")
        
        return specifications
    
    def _build_extraction_prompt(self, text, base_item_id):
        """Build prompt for specification extraction with examples."""
        return f"""
        You are extracting technical specifications from engineering documents. 
        Extract all specifications from the following text related to pumps, valves, and other water/wastewater equipment.

        For each specification you find, extract:
        1. The parameter name (Label)
        2. The value
        3. The unit of measurement (if any)
        4. The type of constraint (exact, minimum, maximum, or range)
        5. Any relevant notes about the specification

        FORMAT EACH SPECIFICATION AS:
        {{
          "item_id": "{base_item_id}", 
          "parameter": "Parameter name",
          "value": value,
          "unit": "unit",
          "constraint_type": "exact/minimum/maximum/range",
          "note": "Any additional details"
        }}

        GROUPING RULES:
        - Specifications for the same component (like a single valve) should share the same item_id
        - Different components get different item_ids by incrementing the number ({base_item_id}, {base_item_id}_2, etc.)
        - Try to infer which specifications belong to the same component

        EXAMPLES:

        Text: "Pump shall have 2-inch inlet and 1.5-inch outlet connections."
        Extracted: 
        {{
          "item_id": "{base_item_id}",
          "parameter": "Pump inlet size",
          "value": 2,
          "unit": "inches",
          "constraint_type": "exact",
          "note": ""
        }}
        {{
          "item_id": "{base_item_id}",
          "parameter": "Pump outlet size",
          "value": 1.5,
          "unit": "inches",
          "constraint_type": "exact",
          "note": ""
        }}

        Text: "Motor shall be minimum 5 HP, 460V, 3-phase."
        Extracted:
        {{
          "item_id": "{base_item_id}_2",
          "parameter": "Motor power",
          "value": 5,
          "unit": "HP",
          "constraint_type": "minimum",
          "note": ""
        }}
        {{
          "item_id": "{base_item_id}_2",
          "parameter": "Motor voltage",
          "value": 460,
          "unit": "V",
          "constraint_type": "exact",
          "note": "3-phase"
        }}

        Now extract all specifications from this text:
        {text}
        
        Respond with ONLY a JSON array of specifications. No additional explanation or text.
        """
    
    def _parse_specifications_from_response(self, response):
        """Parse specifications from LLM response."""
        import re
        import json
        
        try:
            # Find array in response - look for opening [ and closing ]
            match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            # If no array found, try to extract individual JSON objects
            objects = re.findall(r'\{\s*"item_id".*?\}', response, re.DOTALL)
            if objects:
                parsed_objects = []
                for obj_str in objects:
                    try:
                        parsed_objects.append(json.loads(obj_str))
                    except json.JSONDecodeError:
                        continue
                return parsed_objects
            
            return []
        except Exception as e:
            print(f"Error parsing specifications: {e}")
            return []
    
    def _validate_specifications(self, specifications):
        """
        Validate extracted specifications without hallucinating missing data.
        
        Args:
            specifications: List of extracted specifications
            
        Returns:
            List of specifications with flags for issues
        """
        if not specifications:
            return []
            
        validation_prompt = self._build_validation_prompt(specifications)
        validation_response = self.llm.invoke(validation_prompt).content
        
        # Parse validation response to get flagged issues
        flagged_issues = self._parse_flagged_issues(validation_response)
        
        # Mark specifications with flags but don't modify the actual data
        for issue in flagged_issues:
            for spec in specifications:
                if spec["item_id"] == issue["item_id"] and spec["parameter"] == issue["parameter"]:
                    if "flags" not in spec:
                        spec["flags"] = []
                    spec["flags"].append({
                        "issue_type": issue["issue"],
                        "confidence": issue["confidence"],
                        "note": issue["note"]
                    })
        
        return specifications
    
    def _build_validation_prompt(self, specifications):
        """Build prompt for validating specifications."""
        import json
        specs_json = json.dumps(specifications, indent=2)
        
        return f"""
        Review these extracted specifications for accuracy and completeness. 
        DO NOT invent or hallucinate any information not explicitly stated in the original text.

        For each specification, flag any of these issues:
        1. MISSING_UNIT: Value has no unit but should have one
        2. AMBIGUOUS_CONSTRAINT: Not clear if value is exact, minimum, maximum, or range
        3. INCOMPLETE_VALUE: Value appears to be missing digits or parts
        4. UNCERTAIN_GROUPING: Not clear if this specification belongs with its assigned item_id

        Specifications to review:
        {specs_json}

        Format your response as:
        {{
          "flagged_items": [
            {{
              "item_id": "Item_XXX",
              "parameter": "Parameter name",
              "issue": "ISSUE_TYPE",
              "confidence": 0.7,
              "note": "Explanation of the issue, NO suggestions for filling missing data"
            }}
          ]
        }}

        The confidence value (0-1) represents how certain you are that this is an issue.
        """
    
    def _parse_flagged_issues(self, response):
        """Parse flagged issues from validation response."""
        import json
        import re
        
        try:
            # Look for JSON object in response
            match = re.search(r'\{\s*"flagged_items".*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return data.get("flagged_items", [])
            return []
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return []
    
    def save_specifications(self, specifications, document_id=None):
        """
        Save specifications to JSON file.
        
        Args:
            specifications: List of extracted specifications
            document_id: Optional ID for the document
            
        Returns:
            Path to saved JSON file
        """
        if not specifications:
            return None
            
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"specs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create data structure
        data = {
            "document_id": document_id,
            "extraction_date": datetime.now().isoformat(),
            "specifications": specifications
        }
        
        # Save to file
        file_path = os.path.join(self.storage_dir, f"{document_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return file_path
    
    def load_specifications(self, document_id):
        """
        Load specifications from JSON file.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of specifications
        """
        file_path = os.path.join(self.storage_dir, f"{document_id}.json")
        
        if not os.path.exists(file_path):
            return []
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data.get("specifications", [])
        except Exception as e:
            print(f"Error loading specifications: {e}")
            return []
    
    def get_all_document_ids(self):
        """
        Get all document IDs with saved specifications.
        
        Returns:
            List of document IDs
        """
        try:
            files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
            return [os.path.splitext(f)[0] for f in files]
        except Exception as e:
            print(f"Error getting document IDs: {e}")
            return []
```

### 3. Update Embeddings Module

Modify the existing document loading functions to preserve page numbers:

```python
# In embeddings.py

def load_and_split_documents_hierarchically(docs_directory, llm, status_callback=None):
    # Existing code...
    
    # Process each document hierarchically
    all_chunks = []
    for i, document in enumerate(documents):
        # Ensure proper page numbering for PDFs (convert from 0-indexed to 1-indexed)
        if document.metadata.get("source", "").lower().endswith(".pdf"):
            if "page" in document.metadata:
                document.metadata["page_number"] = document.metadata["page"] + 1
            
        # Rest of existing code...
```

### 4. Add Specification Processing to App.py

Add specification extraction during document processing:

```python
# In app.py

# Add import
from specification_manager import SpecificationManager

# Initialize in session state if not already there
if "specification_manager" not in st.session_state:
    st.session_state.specification_manager = None

if "current_specifications" not in st.session_state:
    st.session_state.current_specifications = []

# In document processing section:
if uploaded_files and (process_temp or archive_upload):
    # Existing code for document processing...
    
    # After processing documents:
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
            
            status_text.text(f"Extracted {len(specifications)} specifications from documents")
```

### 5. Update Analysis Tab UI

Update the Analysis tab to display the specifications table automatically:

```python
# In app.py, Analysis tab section

with tab2:
    st.header("Document and Product Analysis")
    
    # Create tabs for different analysis views
    analysis_tabs = st.tabs(["Specifications", "Product Matches", "Document Structure"])
    
    with analysis_tabs[0]:  # Specifications tab
        st.subheader("Extracted Specifications")
        
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
                import pandas as pd
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
        
        if st.session_state.current_specifications and st.session_state.current_specifications:
            # Match specifications to products
            if "product_matches" not in st.session_state or not st.session_state.product_matches:
                with st.spinner("Matching specifications to products..."):
                    # Initialize product matcher
                    from product_matcher import ProductMatcher
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
```

### 6. Add "Scan Documents" Button to Analysis Tab

Add a button to scan documents for specifications on demand:

```python
# In app.py, Analysis tab

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
```

### 7. Update Product Matcher to Support Batch Matching

Enhance the product matcher to efficiently handle batches of specifications:

```python
# In product_matcher.py

def batch_match_specifications(self, specifications, batch_size=10):
    """
    Match specifications to products in batches for efficiency.
    
    Args:
        specifications: List of specifications to match
        batch_size: Number of specifications to process in each batch
        
    Returns:
        List of matching products with scores
    """
    all_matches = []
    
    # Process specifications in batches
    for i in range(0, len(specifications), batch_size):
        batch = specifications[i:i+batch_size]
        batch_matches = self.match_specifications(batch)
        all_matches.extend(batch_matches)
    
    # Deduplicate matches by product and company
    unique_matches = {}
    for match in all_matches:
        key = f"{match['company']}_{match['product']}"
        if key not in unique_matches or match['score'] > unique_matches[key]['score']:
            unique_matches[key] = match
    
    # Sort by score
    result = list(unique_matches.values())
    result.sort(key=lambda x: x['score'], reverse=True)
    
    return result
```

## Integration Testing

After implementing the changes, test with the following scenarios:

1. **Basic Upload Test**:
   - Upload a simple PDF document with clear specifications
   - Verify page numbers are correctly extracted
   - Check that specifications are correctly identified and grouped

2. **Multi-Document Test**:
   - Upload multiple documents with related specifications
   - Verify specifications from different documents are properly separated
   - Check that the combined table shows all specifications with correct sources

3. **Quality Control Test**:
   - Upload a document with ambiguous or incomplete specifications
   - Verify that issues are correctly flagged
   - Check that flagged issues display properly in the UI

4. **Product Matching Test**:
   - Upload a document with specifications matching products in your portfolio
   - Check that product matches are correctly identified
   - Verify that match scores and reasoning are accurate

## Code Structure Summary

The implementation involves:

1. **New Files**:
   - `specification_manager.py`: Core logic for specification extraction and storage

2. **Modified Files**:
   - `embeddings.py`: Fix page number extraction
   - `app.py`: Integration with UI and processing logic
   - `product_matcher.py`: Enhanced batch matching

3. **New Functionality**:
   - Automatic specification extraction during document upload
   - Specification quality validation
   - Analysis tab with specification table display
   - On-demand document scanning for specifications
   - CSV export of specifications

This approach maintains compatibility with existing functionality while adding the new specification table generation feature.
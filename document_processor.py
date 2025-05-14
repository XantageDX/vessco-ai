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
        
        # If no headers found, create a single section for the entire document
        if not potential_headers:
            return [{
                "text": text,
                "header": filename,
                "level": 1,
                "position": 0
            }]
        
        # Sort by position in document
        potential_headers.sort()
        
        # Second pass: use LLM to validate headers and determine hierarchy
        sections = []
        for i, (pos, header_text) in enumerate(potential_headers):
            next_pos = potential_headers[i+1][0] if i < len(potential_headers)-1 else len(text)
            section_text = text[pos:next_pos]
            
            # Skip extremely short sections (likely false positives)
            if len(section_text.strip()) < 50:
                continue
            
            # Use LLM to validate this is a real section header
            # and determine its level in the hierarchy
            try:
                prompt = """
                Analyze this potential document section header and content:
                
                HEADER: """ + header_text + """
                
                CONTENT PREVIEW: """ + section_text[:500] + """...
                
                Is this a valid section header in a technical specification document?
                If yes, what level in the hierarchy is it (1=highest, 5=lowest)?
                Respond with only: VALID,NUMBER where NUMBER is the level as a digit 1-5
                Or respond with: INVALID
                """
                
                response = self.llm.invoke(prompt).content.strip()
                
                if response.startswith("VALID"):
                    try:
                        parts = response.split(",")
                        level_str = parts[1].strip() if len(parts) > 1 else "1"
                        # Extract just the digit from the level string
                        level_digit = ''.join(c for c in level_str if c.isdigit())
                        level = int(level_digit) if level_digit else 1
                        sections.append({
                            "text": section_text,
                            "header": header_text,
                            "level": level,
                            "position": pos
                        })
                    except Exception as e:
                        print(f"Error processing header level: {e}. Using default level 1.")
                        sections.append({
                            "text": section_text,
                            "header": header_text,
                            "level": 1,  # Default to level 1 if parsing fails
                            "position": pos
                        })
            except Exception as e:
                print(f"Error processing header: {header_text}. Error: {e}")
                # Add the section with default level 1 even if we encounter an error
                sections.append({
                    "text": section_text,
                    "header": header_text,
                    "level": 1,
                    "position": pos
                })
        
        # If no valid sections were found, create a single section for the entire document
        if not sections:
            sections.append({
                "text": text,
                "header": filename,
                "level": 1,
                "position": 0
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
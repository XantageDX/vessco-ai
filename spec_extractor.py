# # spec_extractor.py
# from typing import List, Dict, Any
# import json
# import re

# class SpecificationExtractor:
#     """Extract structured specifications from technical documents."""
    
#     def __init__(self, llm):
#         self.llm = llm
    
#     def extract_specifications(self, text: str) -> List[Dict[str, Any]]:
#         """
#         Extract key specifications from text.
#         Returns a list of structured specification objects.
#         """
#         try:
#             # Trim text if it's very long to avoid LLM context limitations
#             max_text_length = 8000  # Adjust based on your LLM's context window
#             if len(text) > max_text_length:
#                 text = text[:max_text_length] + "..."
            
#             prompt = f"""
#             Extract technical specifications from the following text. Focus on:
#             - Dimensions (size, capacity, flow rates)
#             - Materials
#             - Performance requirements
#             - Compliance standards
#             - Operating conditions
            
#             Format each specification as a JSON object with these fields:
#             - parameter: The parameter being specified
#             - value: The required value or range
#             - unit: The unit of measurement (if applicable)
#             - constraint_type: "minimum", "maximum", "exact", or "range"
            
#             If no specifications are found, return an empty array.
            
#             TEXT:
#             {text}
            
#             SPECIFICATIONS (JSON array):
#             """
            
#             response = self.llm.invoke(prompt).content
            
#             # Parse the response to get specifications
#             try:
#                 # Find JSON in the response
#                 start_idx = response.find("[")
#                 end_idx = response.rfind("]") + 1
#                 if start_idx >= 0 and end_idx > start_idx:
#                     json_str = response[start_idx:end_idx]
#                     specifications = json.loads(json_str)
#                     return specifications
#                 return []
#             except Exception as e:
#                 print(f"Error parsing specifications JSON: {e}")
#                 # Try a more robust parsing approach
#                 return self._extract_with_regex(response)
#         except Exception as e:
#             print(f"Error extracting specifications: {e}")
#             return []
    
#     def _extract_with_regex(self, response: str) -> List[Dict[str, Any]]:
#         """
#         Fallback method to extract specifications using regex if JSON parsing fails.
#         """
#         specifications = []
        
#         # Look for patterns that might represent specifications
#         # Parameter: value unit (constraint)
#         spec_pattern = r'"parameter":\s*"([^"]+)",\s*"value":\s*"?([^",]+)"?,\s*"unit":\s*"?([^",]*)"?,\s*"constraint_type":\s*"([^"]+)"'
        
#         matches = re.finditer(spec_pattern, response)
#         for match in matches:
#             param, value, unit, constraint = match.groups()
            
#             # Try to convert value to number if possible
#             try:
#                 if '.' in value:
#                     value = float(value)
#                 else:
#                     value = int(value)
#             except ValueError:
#                 # Keep as string if not a number
#                 pass
                
#             specifications.append({
#                 "parameter": param.strip(),
#                 "value": value,
#                 "unit": unit.strip(),
#                 "constraint_type": constraint.strip()
#             })
            
#         return specifications

#### WITH SOURCE/METADATA COLUMNS ####
# spec_extractor.py
from typing import List, Dict, Any
import json
import re

class SpecificationExtractor:
    """Extract structured specifications from technical documents."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_specifications(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
        """
        Extract key specifications from text.
        Returns a list of structured specification objects.
        
        Args:
            text: The text to extract specifications from
            metadata: Document metadata including source document and page information
        """
        try:
            # Get source information from metadata
            source_info = {}
            if metadata:
                source_doc = metadata.get("source", "Unknown source")
                page_numbers = []
                
                # Handle different page metadata formats
                if "page" in metadata:
                    page_numbers.append(str(metadata["page"]))
                elif "page_number" in metadata:
                    page_numbers.append(str(metadata["page_number"]))
                
                source_info = {
                    "source_document": source_doc,
                    "page_numbers": page_numbers,
                }
            
            # Trim text if it's very long to avoid LLM context limitations
            max_text_length = 8000  # Adjust based on your LLM's context window
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
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
            
            If no specifications are found, return an empty array.
            
            TEXT:
            {text}
            
            SPECIFICATIONS (JSON array):
            """
            
            response = self.llm.invoke(prompt).content
            
            # Parse the response to get specifications
            try:
                # Find JSON in the response
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    specifications = json.loads(json_str)
                    
                    # Add source information to each specification
                    for spec in specifications:
                        spec.update(source_info)
                    
                    return specifications
                return []
            except Exception as e:
                print(f"Error parsing specifications JSON: {e}")
                # Try a more robust parsing approach
                specs = self._extract_with_regex(response)
                
                # Add source information to each specification
                for spec in specs:
                    spec.update(source_info)
                
                return specs
        except Exception as e:
            print(f"Error extracting specifications: {e}")
            return []
    
    def _extract_with_regex(self, response: str) -> List[Dict[str, Any]]:
        """
        Fallback method to extract specifications using regex if JSON parsing fails.
        """
        specifications = []
        
        # Look for patterns that might represent specifications
        # Parameter: value unit (constraint)
        spec_pattern = r'"parameter":\s*"([^"]+)",\s*"value":\s*"?([^",]+)"?,\s*"unit":\s*"?([^",]*)"?,\s*"constraint_type":\s*"([^"]+)"'
        
        matches = re.finditer(spec_pattern, response)
        for match in matches:
            param, value, unit, constraint = match.groups()
            
            # Try to convert value to number if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if not a number
                pass
                
            specifications.append({
                "parameter": param.strip(),
                "value": value,
                "unit": unit.strip(),
                "constraint_type": constraint.strip()
            })
            
        return specifications
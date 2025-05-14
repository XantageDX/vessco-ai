# product_matcher.py
import json
from typing import List, Dict, Any
import re
import os

class ProductMatcher:
    """Match extracted specifications to product portfolio."""
    
    def __init__(self, llm, portfolio_path="portfolio.json"):
        self.llm = llm
        self.portfolio_path = portfolio_path
        self.portfolio = self._load_portfolio()
    
    def _load_portfolio(self):
        """Load the product portfolio from JSON file."""
        if not os.path.exists(self.portfolio_path):
            print(f"Warning: Portfolio file {self.portfolio_path} not found")
            return {"portfolio": []}
            
        try:
            with open(self.portfolio_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return {"portfolio": []}
    
    def reload_portfolio(self):
        """Reload the portfolio from disk."""
        self.portfolio = self._load_portfolio()
    
    def match_specifications(self, specifications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match specifications to products in the portfolio.
        Returns a list of matching products with confidence scores.
        """
        if not specifications:
            return []
        
        try:
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
                            
                            try:
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
                                        "reasoning": reasoning,
                                        "description": product_description,
                                        "use_case": use_case
                                    })
                                else:
                                    # Fallback if no score is found
                                    print(f"Could not parse score for {company} - {product_name}, using default")
                                    matches.append({
                                        "company": company,
                                        "product": product_name,
                                        "score": 0,
                                        "reasoning": "Could not evaluate match properly.",
                                        "description": product_description,
                                        "use_case": use_case
                                    })
                            except Exception as e:
                                print(f"Error matching product {company} - {product_name}: {e}")
                                # Skip this product and continue
                                continue
            
            # Sort by score (highest first)
            matches.sort(key=lambda x: x["score"], reverse=True)
            return matches
            
        except Exception as e:
            print(f"Error in match_specifications: {e}")
            return []
        
    def get_product_details(self, company, product_name):
        """
        Get full details for a specific product.
        """
        for product_entry in self.portfolio.get("portfolio", []):
            if company in product_entry:
                for product_type in product_entry[company].get("Tipes", []):
                    if product_name in product_type:
                        return {
                            "company": company,
                            "product": product_name,
                            "description": product_type[product_name],
                            "use_case": product_type.get("Use Case", "")
                        }
        return None
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
    
    def save_portfolio(self, updated_portfolio=None):
        """Save portfolio data back to file."""
        try:
            with open(self.portfolio_path, 'w') as f:
                json.dump(updated_portfolio or self.portfolio, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving portfolio: {e}")
            return False
    
    def add_product(self, company, product_name, description, use_case=""):
        """Add a new product to the portfolio."""
        # Find the company if it exists
        company_found = False
        for product_entry in self.portfolio.get("portfolio", []):
            if company in product_entry:
                company_found = True
                # Add product to existing company
                product_entry[company].get("Tipes", []).append({
                    product_name: description,
                    "Use Case": use_case
                })
                break
        
        # If company not found, create a new entry
        if not company_found:
            self.portfolio.get("portfolio", []).append({
                company: {
                    "Tipes": [
                        {
                            product_name: description,
                            "Use Case": use_case
                        }
                    ],
                    "Description": f"Products by {company}"
                }
            })
        
        # Save changes
        return self.save_portfolio()
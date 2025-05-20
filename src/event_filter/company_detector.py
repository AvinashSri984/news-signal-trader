import re
from typing import List, Dict, Set, Optional
import json
import os
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CompanyDetector:
    def __init__(self):
        """Initialize the company detector with enhanced matching capabilities"""
        try:
            self.companies = self._load_company_data()
            self.alternative_names = self._build_alternative_names()
            self.ticker_to_company = self._build_ticker_map()
            self.name_to_company = self._build_name_map()
            
            # Initialize matching patterns
            self.possessive_pattern = re.compile(r"'s\b")
            self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
            
            logger.info("CompanyDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CompanyDetector: {str(e)}")
            raise

    def _load_company_data(self) -> Dict:
        """Load company data from JSON file"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), 'company_data.json')
            with open(data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading company data: {str(e)}")
            return {}

    def _build_alternative_names(self) -> Dict[str, str]:
        """Build mapping of alternative names to company names"""
        try:
            alternatives = {}
            for company, data in self.companies.items():
                # Add company name
                alternatives[company.lower()] = company
                
                # Add alternative names
                for alt_name in data.get('alternative_names', []):
                    alternatives[alt_name.lower()] = company
                    
                # Add common variations
                base_name = company.lower()
                alternatives[base_name.replace('&', 'and')] = company
                alternatives[base_name.replace('and', '&')] = company
                
            return alternatives
        except Exception as e:
            logger.error(f"Error building alternative names: {str(e)}")
            return {}

    def _build_ticker_map(self) -> Dict[str, str]:
        """Build mapping of ticker symbols to company names"""
        try:
            return {data['ticker'].lower(): company 
                   for company, data in self.companies.items()}
        except Exception as e:
            logger.error(f"Error building ticker map: {str(e)}")
            return {}

    def _build_name_map(self) -> Dict[str, str]:
        """Build mapping of company names to their data"""
        try:
            return {company.lower(): company for company in self.companies.keys()}
        except Exception as e:
            logger.error(f"Error building name map: {str(e)}")
            return {}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove possessive forms
            text = self.possessive_pattern.sub('', text)
            
            # Remove special characters
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return text

    def _find_ticker_matches(self, text: str) -> Set[str]:
        """Find companies by ticker symbols"""
        try:
            matches = set()
            for match in self.ticker_pattern.finditer(text):
                ticker = match.group().lower()
                if ticker in self.ticker_to_company:
                    matches.add(self.ticker_to_company[ticker])
            return matches
        except Exception as e:
            logger.error(f"Error finding ticker matches: {str(e)}")
            return set()

    def _find_name_matches(self, text: str) -> Set[str]:
        """Find companies by name matches"""
        try:
            matches = set()
            words = text.split()
            
            # Check for exact matches
            for i in range(len(words)):
                for j in range(i + 1, min(i + 5, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    if phrase in self.name_to_company:
                        matches.add(self.name_to_company[phrase])
                    if phrase in self.alternative_names:
                        matches.add(self.alternative_names[phrase])
            
            return matches
        except Exception as e:
            logger.error(f"Error finding name matches: {str(e)}")
            return set()

    def detect_companies(self, text: str) -> List[Dict]:
        """Detect companies mentioned in text with enhanced matching"""
        try:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Find matches
            ticker_matches = self._find_ticker_matches(normalized_text)
            name_matches = self._find_name_matches(normalized_text)
            
            # Combine matches
            all_matches = ticker_matches.union(name_matches)
            
            # Prepare results
            results = []
            for company in all_matches:
                company_data = self.companies[company].copy()
                company_data['name'] = company
                results.append(company_data)
            
            logger.info(f"Detected {len(results)} companies in text")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting companies: {str(e)}")
            return []

    def get_company_info(self, company: str) -> Optional[Dict]:
        """Get detailed company information"""
        try:
            if company in self.companies:
                company_data = self.companies[company].copy()
                company_data['name'] = company
                return company_data
            return None
        except Exception as e:
            logger.error(f"Error getting company info: {str(e)}")
            return None

    def get_ticker(self, company_name: str) -> str:
        """
        Get the ticker symbol for a given company name.
        
        Args:
            company_name (str): The name of the company
            
        Returns:
            str: The ticker symbol if found, None otherwise
        """
        return self.ticker_to_company.get(company_name.lower()) 
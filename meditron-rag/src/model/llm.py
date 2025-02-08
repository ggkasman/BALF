"""
LLM module for interacting with Hugging Face Endpoint.
"""
import os
import logging
import asyncio
import aiohttp
import requests
import random
import re
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic.v1 import Field

from config.settings import MEDITRON_ENDPOINT_URL

logger = logging.getLogger(__name__)

class MeditronLLM(LLM):
    """LangChain wrapper for Hugging Face Endpoint (llama.cpp format)."""
    
    endpoint_url: str = Field(default_factory=lambda: MEDITRON_ENDPOINT_URL)
    api_token: str = Field(default_factory=lambda: os.getenv("HUGGINGFACE_API_TOKEN"))
    completion_url: str = Field(default="")
    max_retries: int = Field(default=3)
    timeout: int = Field(default=180)  # Increased timeout for T4 GPU
    connection_timeout: int = Field(default=30)  # Separate timeout for initial connection
    
    def __init__(self, **kwargs):
        """Initialize the Meditron LLM."""
        super().__init__(**kwargs)
        
        logger.info("Initializing MeditronLLM...")
        
        if not self.endpoint_url:
            logger.error("MEDITRON_ENDPOINT_URL not found in environment variables")
            raise ValueError(
                "MEDITRON_ENDPOINT_URL not found in environment variables. "
                "Please set it in your .env file."
            )
            
        if not self.api_token:
            logger.error("HUGGINGFACE_API_TOKEN not found in environment variables")
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not found in environment variables. "
                "Please set it in your .env file."
            )
            
        # Clean up the endpoint URL and add completion path
        self.endpoint_url = self.endpoint_url.rstrip('/')
        self.completion_url = f"{self.endpoint_url}/completion"
        logger.info(f"Using Hugging Face endpoint: {self.completion_url}")
        
        # Test the endpoint with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Testing endpoint connection (attempt {attempt + 1}/{self.max_retries})...")
                headers = {"Authorization": f"Bearer {self.api_token}"}
                payload = {
                    "prompt": "What is bronchoalveolar lavage?",  # Simple medical test query
                    "temperature": 0.1,
                    "max_tokens": 100
                }
                logger.info("Sending test query to endpoint...")
                response = requests.post(
                    self.completion_url,
                    headers=headers,
                    json=payload,
                    timeout=self.connection_timeout  # Use shorter timeout for connection test
                )
                response.raise_for_status()
                test_result = response.json()
                logger.info(f"Test query response type: {type(test_result)}")
                logger.info(f"Test query raw response: {test_result}")
                logger.info("Successfully connected to endpoint")
                break
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to connect to endpoint after {self.max_retries} attempts: {str(e)}")
                    logger.error("Please check:")
                    logger.error("1. Your Hugging Face API token is correct")
                    logger.error("2. The endpoint URL is correct")
                    logger.error("3. The endpoint is running and accessible")
                    raise
                else:
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                    continue
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "huggingface-endpoint"
    
    def _format_text(self, text: str) -> str:
        """Format and clean text to handle common issues."""
        if not text:
            return ""
            
        # Fix hyphenation and spaces
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Remove hyphenation with spaces
        text = re.sub(r'(\w)-\n+(\w)', r'\1\2', text)  # Remove hyphenation with newlines
        text = re.sub(r'(?<=\w)\s+(?=[.,!?])', '', text)  # Remove spaces before punctuation
        
        # Fix common medical terms
        replacements = {
            r'fl\s*u\s*i\s*d': 'fluid',
            r'al-?\s*ve\s*o\s*li': 'alveoli',
            r'br\s*on\s*ch\s*o': 'broncho',
            r'al\s*ve\s*o\s*lar': 'alveolar',
            r'la\s*v\s*age': 'lavage',
            r'mac\s*ro\s*phages?': 'macrophages',
            r'ly\s*mp\s*h\s*o\s*cytes?': 'lymphocytes',
            r'ne\s*ut\s*ro\s*phils?': 'neutrophils',
            r'eo\s*si\s*no\s*phils?': 'eosinophils'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix spacing around numbers and units
        text = re.sub(r'(\d)\s*%', r'\1%', text)  # Remove space between number and percent
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Join split numbers
        text = re.sub(r'(\d),\s*(\d)', r'\1.\2', text)  # Convert European decimals to US
        
        # Fix common formatting issues
        text = re.sub(r'\s*\n\s*', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        text = re.sub(r'\.+', '.', text)  # Collapse multiple periods
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)  # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
        
        # Fix capitalization
        text = re.sub(r'(?<=\. )[a-z]', lambda m: m.group().upper(), text)  # Capitalize after periods
        text = text.strip()
        
        return text

    def _clean_response(self, response: str) -> str:
        """Clean and format the response."""
        if not response:
            return ""
            
        # Remove any XML-like tags and their content
        response = re.sub(r'</?\w+>', '', response)  # Remove XML tags
        
        # Fix spaces in numbers and percentages
        response = re.sub(r'(\d)\s*%', r'\1%', response)  # Fix percentage spacing
        response = re.sub(r'(\d)\s+(\d)', r'\1\2', response)  # Join split numbers
        
        # Fix common medical terms with better patterns
        medical_terms = {
            r'\bfl\s*u\s*i\s*d\b': 'fluid',
            r'\bb\s*a\s*l\b': 'BAL',
            r'\bly\s*[ms]\s*[mph]\s*[ho]?\s*c\s*y\s*t\s*e\s*s?\b': 'lymphocytes',
            r'\bn\s*e\s*u\s*t\s*r\s*o\s*ph\s*i\s*l\s*s?\b': 'neutrophils',
            r'\be\s*o\s*s\s*i\s*n\s*o\s*ph\s*i\s*l\s*s?\b': 'eosinophils',
            r'\bm\s*a\s*c\s*r\s*o\s*ph\s*a\s*g\s*e\s*s?\b': 'macrophages',
            r'\bl\s*e\s*u\s*k\s*o\s*c\s*y\s*t\s*e\s*s?\b': 'leukocytes',
            r'\br\s*a\s*t\s*i\s*o\s*s?\b': 'ratios'
        }
        
        for pattern, replacement in medical_terms.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        # Fix common formatting issues
        response = re.sub(r'\s*,\s*', ', ', response)  # Fix comma spacing
        response = re.sub(r'\s+', ' ', response)  # Normalize spaces
        response = re.sub(r'(?<=\d)\s*-\s*(?=\d)', '-', response)  # Fix range dashes
        
        # Fix specific number issues
        response = re.sub(r'(?<=\d)\.(?=\d)', '.', response)  # Fix decimal points
        response = response.replace('I', '1')  # Replace common OCR error
        
        # Split into sentences and clean each one
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        cleaned_sentences = []
        
        for sentence in sentences:
            # Capitalize first letter
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            # Remove extra spaces
            sentence = re.sub(r'\s+', ' ', sentence)
            if len(sentence) > 3:  # Only keep meaningful sentences
                cleaned_sentences.append(sentence)
        
        # Join sentences with proper punctuation
        response = '. '.join(cleaned_sentences)
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response.strip()

    def _validate_response(self, response: str) -> bool:
        """Validate that the response is relevant and not stuck in a loop."""
        if not response:
            return False
            
        # Only reject extremely short responses
        if len(response.strip()) < 3:
            logger.warning("Response too short")
            return False
            
        # Check for irrelevant content
        response_lower = response.lower()
        irrelevant_terms = ['game', 'xbox', 'nintendo', 'wii', 'console', 'pc']
        if all(term in response_lower for term in irrelevant_terms):  # Only reject if ALL terms are present
            logger.warning("Response appears to be completely off-topic")
            return False
            
        # Accept almost everything else
        return True

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to the Hugging Face endpoint."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Preparing API call (attempt {attempt + 1}/{self.max_retries})...")
                logger.info(f"Prompt length: {len(prompt)} characters")
                
                # Prepare the request with llama.cpp format
                headers = {"Authorization": f"Bearer {self.api_token}"}
                payload = {
                    "prompt": prompt,
                    "temperature": 0.1,
                    "max_tokens": 3072,  # Decreased from 4096 for more efficient responses while maintaining good length
                    "top_p": 0.2,
                    "repeat_penalty": 2.0,
                    "stop": stop or [
                        "QUESTION:", "\nQUESTION", "ANSWER:", "\nANSWER",
                        "</context>", "<context>", "\n\n\n",
                        "</answer>", "</answers>", "References:", "\nReferences",
                        "Citation:", "\nCitation", "\n\n[", "\n[", "[1]",
                        "1.", "2.", "3.", "(1)", "(2)", "(3)",
                        "doi:", "http", "©", ";", "–"
                    ],
                    "stream": False,
                    "frequency_penalty": 1.2,
                    "presence_penalty": 1.2,
                    "mirostat_mode": 2,
                    "mirostat_tau": 5.0,
                    "mirostat_eta": 0.1,
                    "min_p": 0.1,
                    "typical_p": 0.4,
                    "tfs_z": 1.0,
                    "top_k": 30
                }
                
                # Make the async request
                logger.info("Calling endpoint...")
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.completion_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        # Extract the response text
                        response_text = ""
                        if isinstance(result, dict):
                            # Try to get the response from various possible keys
                            for key in ["content", "answer", "generated_text", "text", "response"]:
                                if key in result and result[key]:
                                    response_text = result[key].strip()
                                    break
                                    
                            if not response_text and "choices" in result and result["choices"]:
                                choices = result["choices"]
                                if isinstance(choices, list) and choices:
                                    for choice in choices:
                                        if isinstance(choice, dict):
                                            for key in ["text", "content", "message"]:
                                                if key in choice and choice[key]:
                                                    response_text = choice[key].strip()
                                                    break
                                        elif isinstance(choice, str):
                                            response_text = choice.strip()
                                            break
                                    
                        elif isinstance(result, str):
                            response_text = result.strip()
                            
                        if response_text:
                            # Clean and format the response
                            cleaned_response = self._clean_response(response_text)
                            if cleaned_response:
                                return cleaned_response
                            
                        # If we got no response, try again
                        continue
                
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise last_error
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous call to the Hugging Face endpoint."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Preparing API call (attempt {attempt + 1}/{self.max_retries})...")
                
                # Prepare the request with llama.cpp format
                headers = {"Authorization": f"Bearer {self.api_token}"}
                payload = {
                    "prompt": prompt,
                    "temperature": 0.1,  # Keep low temperature for deterministic output
                    "max_tokens": 1024,  # Increased from 512 for more detailed responses
                    "top_p": 0.2,  # Reduced from 0.1 for more focused output
                    "repeat_penalty": 2.0,  # Increased from 1.8 for less repetition
                    "stop": [
                        "QUESTION:", "\nQUESTION", "ANSWER:", "\nANSWER",
                        "</context>", "<context>", "\n\n\n",
                        "</answer>", "</answers>", "References:", "\nReferences",
                        "Citation:", "\nCitation", "\n\n[", "\n[", "[1]",
                        "1.", "2.", "3.", "(1)", "(2)", "(3)",
                        "doi:", "http", "©", ";", "–"
                    ],
                    "stream": False,
                    "frequency_penalty": 1.2,  # Increased from 1.0 for more diverse vocabulary
                    "presence_penalty": 1.2,  # Increased from 1.0 for less repetition
                    "mirostat_mode": 2,  # Keep aggressive Mirostat sampling
                    "mirostat_tau": 5.0,  # Increased from 4.0 for more informative content
                    "mirostat_eta": 0.1,  # Keep small learning rate
                    "min_p": 0.1,  # Increased from 0.05 for better coherence
                    "typical_p": 0.4,  # Increased from 0.3 for more natural responses
                    "tfs_z": 1.0,  # Keep tail-free sampling
                    "top_k": 30  # Increased from 20 for more vocabulary diversity
                }
                
                # Make the request
                logger.info("Calling endpoint...")
                response = requests.post(
                    self.completion_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Extract the response
                result = response.json()
                logger.info("Received response from endpoint")
                logger.debug(f"Raw response: {result}")
                
                # Handle llama.cpp response format
                if isinstance(result, dict):
                    if "content" in result:
                        return result["content"].strip()
                    elif "answer" in result:
                        return result["answer"].strip()
                    else:
                        logger.warning(f"Unexpected response format: {result}")
                        return str(result)
                else:
                    logger.warning(f"Unexpected response type: {type(result)}")
                    return str(result)
                
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts: {str(e)}")
                    raise
                else:
                    logger.warning(f"Request attempt {attempt + 1} failed, retrying...")
                    continue
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise last_error
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint_url": self.endpoint_url,
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.95,
            "repeat_penalty": 1.5
        } 
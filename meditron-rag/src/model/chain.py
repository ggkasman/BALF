"""
Chain module for RAG pipeline implementation.
"""
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)

class MeditronChain:
    """RAG chain implementation using Meditron model."""
    
    def __init__(self, llm, vector_store):
        """Initialize the RAG chain.
        
        Args:
            llm: Language model instance
            vector_store: Vector store for document retrieval
        """
        self.llm = llm
        # Configure retriever with uniqueness check
        self.retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 4,  # Number of documents to retrieve
                "filter": None  # No filtering
            }
        )
        
        # Track seen documents to prevent duplicates
        self._seen_docs = set()
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Enhanced prompt with more specific instructions
        self.prompt_prefix = """You are a medical expert specializing in bronchoalveolar lavage (BAL) and related procedures. You will be provided with context information and a specific question. Your task is to answer the question using ONLY the information from the provided context.

INSTRUCTIONS:
1. Start with a clear, direct answer to the question
2. Provide complete, well-structured sentences
3. Include specific numbers and ranges when available
4. Be concise and avoid repetition
5. When discussing BAL results, compare them to normal ranges
6. End with a concluding statement

RULES:
- Answer ONLY using information from the provided context
- Do NOT make up or infer information not present in the context
- Do NOT repeat information
- Do NOT include citations in your answer
- If the context doesn't contain enough information, explicitly state that

<context>
"""
        self.prompt_question = "\n</context>\n\nQUESTION: "
        self.prompt_suffix = "\n\nANSWER:"
        
        logger.info("Initialized Meditron RAG chain")
        
    def _format_references(self, docs: List[Any]) -> str:
        """Format source documents into a references section."""
        references = []
        for i, doc in enumerate(docs[:3], 1):  # Limit to top 3 sources
            source = doc.metadata.get('source', 'Unknown')
            # Clean up source path to just show filename
            source = Path(source).name if isinstance(source, str) else 'Unknown'
            # Use string concatenation instead of f-strings
            references.append("[" + str(i) + "] " + source)
        
        if references:
            return "\n\nReferences:\n" + "\n".join(references)
        return ""
        
    def fix_hyphenation(self, text: str) -> str:
        """Fix hyphenated words and merged terms."""
        # Handle line-break hyphens and multiple hyphens
        text = re.sub(r'(\w+)-(\w+)-(\w+)', r'\1-\2 \3', text)  # Double hyphens
        text = re.sub(r'(\w)-+(\w)', r'\1\2', text)  # Remove redundant hyphens
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Hyphens with spaces
        
        # Fix merged words after hyphen removal
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\w)(of|in|to|the)(\w)', r'\1 \2 \3', text, flags=re.IGNORECASE)
        
        # Normalize whitespace and remove special hyphen characters
        text = text.replace('‐', '-')  # Replace unicode hyphens
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        return text

    def _clean_response(self, response: str) -> str:
        """Clean and format the LLM response while preserving original content."""
        if not response:
            return ""
        
        # Remove XML-like tags
        response = re.sub(r'</?\w+>', '', response)
        
        # Fix spaces in medical terms
        medical_terms = {
            r'fl\s*u\s*i\s*d': 'fluid',
            r'al\s*ve\s*o\s*lar': 'alveolar',
            r'ly\s*[ms]\s*[mph]\s*[ho]?\s*c\s*y\s*t\s*e\s*s?': 'lymphocytes',
            r'ne\s*u\s*t\s*r\s*o\s*ph\s*i\s*l\s*s?': 'neutrophils',
            r'eo\s*s\s*i\s*n\s*o\s*ph\s*i\s*l\s*e?\s*s?': 'eosinophils',
            r'ma\s*c\s*r\s*o\s*ph\s*a\s*g\s*e\s*s?': 'macrophages',
            r'he\s*al\s*t\s*h\s*y?': 'healthy',
            r'b\s*a\s*l': 'BAL'
        }
        
        for pattern, replacement in medical_terms.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        # Fix number formatting
        response = re.sub(r'(\d+)\s*%', r'\1%', response)  # Fix percentage spacing
        response = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', response)  # Fix range spacing
        response = re.sub(r'one\s*%', '1%', response, flags=re.IGNORECASE)  # Convert written numbers
        
        # Fix general formatting
        response = re.sub(r'\s+', ' ', response)  # Normalize spaces
        response = re.sub(r'\s*,\s*', ', ', response)  # Fix comma spacing
        response = re.sub(r'\s*\.\s*', '. ', response)  # Fix period spacing
        
        # Split into sentences and clean each one
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        cleaned_sentences = []
        seen_sentences = set()  # Track unique sentences
        
        for sentence in sentences:
            # Normalize the sentence for comparison (lowercase, remove extra spaces)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            # Only add if we haven't seen this sentence before
            if normalized not in seen_sentences and len(normalized.split()) > 3:  # Ensure meaningful sentences
                seen_sentences.add(normalized)
                # Use the original sentence with proper capitalization
                cleaned_sentence = sentence[0].upper() + sentence[1:] if sentence else ""
                cleaned_sentences.append(cleaned_sentence)
        
        # Join sentences with proper punctuation
        response = '. '.join(cleaned_sentences)
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response.strip()

    def _validate_response(self, response: str, context: str) -> bool:
        """Validate that the response is grounded in the context."""
        if not response:
            return False
            
        # Special handling for image analysis responses
        if "Image Analysis Results:" in context:
            return True
            
        # Regular validation for other responses
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check for medical relevance first
        medical_terms = {'bal', 'fluid', 'cells', 'macrophages', 'lymphocytes', 'neutrophils', 'eosinophils', 
                        'ratio', 'normal', 'range', 'leukocyte', 'cd4', 'cd8'}
        if not any(term in response_lower for term in medical_terms):
            logger.warning("Response lacks relevant medical terminology")
            return False
            
        # Extract numerical values and ranges from response
        number_pattern = r'\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?(?:\s*%)?'
        numbers = re.findall(number_pattern, response)
        
        # If we have numbers, they should appear in context (being more lenient with exact matches)
        if numbers:
            context_numbers = re.findall(number_pattern, context)
            if not any(num in context_numbers for num in numbers):
                numbers_found = False
                # Check for approximate matches (e.g., "80" matching "80-90")
                for num in numbers:
                    num_val = float(re.findall(r'\d+(?:\.\d+)?', num)[0])
                    for ctx_num in context_numbers:
                        if '-' in ctx_num:
                            range_vals = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', ctx_num)]
                            if len(range_vals) == 2 and range_vals[0] <= num_val <= range_vals[1]:
                                numbers_found = True
                                break
                if not numbers_found:
                    logger.warning("Response contains numbers not found in context")
                    return False
        
        # Check for malformed text but be more lenient
        max_word_length = max(len(word) for word in response_lower.split()) if response_lower else 0
        if max_word_length > 50:  # Only reject if words are extremely long
            logger.warning("Response contains extremely long words")
            return False
            
        return True

    def _select_relevant_context(self, question: str, context: str) -> str:
        """Select relevant sentences from context based on question keywords."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Get question keywords (excluding common words)
        common_words = {'what', 'is', 'are', 'the', 'in', 'on', 'at', 'by', 'for', 'to', 'of', 'and', 'or'}
        question_words = set(word.lower() for word in question.split() if word.lower() not in common_words)
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(word.lower() for word in sentence.split())
            # Count matching keywords
            keyword_matches = len(sentence_words.intersection(question_words))
            # Count medical terms
            medical_terms = {'bal', 'bronchoalveolar', 'lavage', 'cells', 'fluid', 'lung', 'transplant', 'leukocytes'}
            term_matches = len(sentence_words.intersection(medical_terms))
            # Calculate total score
            score = keyword_matches * 2 + term_matches
            if score > 0:
                scored_sentences.append((score, sentence))
        
        if not scored_sentences:
            return ""
            
        # Sort by score and combine top sentences
        scored_sentences.sort(reverse=True)
        return ' '.join(sent for _, sent in scored_sentences[:2])

    def _analyze_cell_counts(self, cell_percentages: Dict[str, str]) -> str:
        """Analyze cell count percentages and compare to normal ranges."""
        try:
            analysis = []
            
            # Convert percentages to floats for comparison
            counts = {}
            for k, v in cell_percentages.items():
                try:
                    if isinstance(v, str):
                        # Remove any quotes and spaces
                        clean_value = v.strip('" ')
                        # Remove the % symbol if present
                        if clean_value.endswith('%'):
                            clean_value = clean_value[:-1]
                        # Convert to float, handling any remaining spaces or special characters
                        clean_value = re.sub(r'[^\d.]', '', clean_value)
                        counts[k] = float(clean_value)
                    elif isinstance(v, (int, float)):
                        counts[k] = float(v) * 100
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {v} to float for {k}: {e}")
                    continue
            
            logger.info(f"Converted counts: {counts}")
            
            if not counts:
                return "Unable to analyze cell counts due to invalid format."
            
            # Check each cell type against normal ranges
            if 'macrophages' in counts:
                mac = counts['macrophages']
                if mac < 80:
                    analysis.append(f"Alveolar macrophages are decreased at {mac:.1f}% (normal: 80-90%)")
                elif mac > 90:
                    analysis.append(f"Alveolar macrophages are elevated at {mac:.1f}% (normal: 80-90%)")
            
            if 'lymphocytes' in counts:
                lym = counts['lymphocytes']
                if lym > 15:
                    analysis.append(f"Lymphocytes are elevated at {lym:.1f}% (normal: 5-15%)")
                elif lym < 5:
                    analysis.append(f"Lymphocytes are decreased at {lym:.1f}% (normal: 5-15%)")
            
            if 'neutrophils' in counts:
                neu = counts['neutrophils']
                if neu > 3:
                    analysis.append(f"Neutrophils are elevated at {neu:.1f}% (normal: 1-3%)")
                elif neu < 1:
                    analysis.append(f"Neutrophils are decreased at {neu:.1f}% (normal: 1-3%)")
            
            if 'eosinophils' in counts:
                eos = counts['eosinophils']
                if eos > 1:
                    analysis.append(f"Eosinophils are elevated at {eos:.1f}% (normal: ≤1%)")
            
            if not analysis:
                return "All cell counts are within normal ranges."
            
            return " ".join(analysis) + "\n\nThese findings may suggest an inflammatory or immune response in the lungs. Further clinical correlation is recommended."
            
        except Exception as e:
            logger.error(f"Error analyzing cell counts: {e}, values: {cell_percentages}")
            return "Unable to analyze cell counts due to invalid format."

    def _truncate_context(self, context: str, max_chars: int = 6000) -> str:
        """Truncate context while preserving complete sentences and relevance."""
        if len(context) <= max_chars:
            return context
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Score sentences by relevance to BAL and medical content
        relevant_terms = {
            'bal': 3, 'bronchoalveolar': 3, 'lavage': 3,
            'fluid': 2, 'cells': 2, 'normal': 2,
            'macrophages': 2, 'lymphocytes': 2, 'neutrophils': 2, 'eosinophils': 2,
            'lung': 1, 'transplant': 1, 'rejection': 1, 'infection': 1,
            'inflammation': 1, 'diagnosis': 1, 'treatment': 1, 'patient': 1
        }
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = set(sentence.lower().split())
            # Calculate weighted term score
            term_score = sum(relevant_terms.get(word, 0) for word in words)
            # Add position bias (favor earlier sentences slightly)
            position_score = 1.0 / (1 + i * 0.1)
            # Calculate length normalized score
            length_score = min(1.0, 50.0 / len(sentence)) if len(sentence) > 0 else 0
            # Combine scores
            total_score = term_score * position_score * length_score
            scored_sentences.append((total_score, sentence))
        
        # Sort by score (descending)
        scored_sentences.sort(reverse=True)
        
        # Build truncated context from most relevant sentences
        result = []
        current_length = 0
        
        for _, sentence in scored_sentences:
            if current_length + len(sentence) + 1 <= max_chars:
                result.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        # Resort sentences in original order for better coherence
        result.sort(key=lambda x: sentences.index(x))
        
        # Join sentences and ensure proper spacing
        final_text = ' '.join(result)
        
        # Add ellipsis if truncated
        if len(final_text) < len(context):
            final_text += "..."
        
        return final_text

    def _clean_document_content(self, content: str) -> str:
        """Clean document content to remove noise and improve formatting."""
        if not content:
            return ""
            
        # Remove citations and references
        content = re.sub(r'\[\d+[,\s]*\d*\]', '', content)  # Remove citation brackets
        content = re.sub(r'\(\d{4}\)', '', content)  # Remove year citations
        content = re.sub(r'\d+;\d+:\d+[-–]\d+', '', content)  # Remove journal citations
        content = re.sub(r'(?i)references?:?\s*$.*', '', content, flags=re.DOTALL)
        
        # Remove headers, footers, and metadata while preserving important content
        content = re.sub(r'(?i)^.*?(?:abstract|introduction):', '', content)
        content = re.sub(r'©.*?reserved\.', '', content)
        content = re.sub(r'doi:.*?(?=\s|$)', '', content)
        content = re.sub(r'http[s]?://\S+', '', content)
        
        # Normalize whitespace and line breaks
        content = re.sub(r'\s*\n\s*\n\s*', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters while preserving medical symbols
        content = re.sub(r'[^\w\s.,;:?!()\-\'\"≤≥±×°%]+', ' ', content)
        
        # Split into sentences for cleaning
        sentences = re.split(r'(?<=[.!?])\s+', content)
        cleaned_sentences = []
        
        for sentence in sentences:
            # Skip very short or incomplete sentences
            if len(sentence.split()) < 3:
                continue
            # Skip sentences that don't start with capital letter or number
            if not sentence[0].isupper() and not sentence[0].isdigit():
                continue
            # Skip sentences that don't end properly
            if not sentence.strip().endswith(('.', '!', '?')):
                continue
            cleaned_sentences.append(sentence.strip())
        
        # Join cleaned sentences
        return ' '.join(cleaned_sentences)

    def _extract_cell_percentages_from_text(self, text: str) -> Dict[str, str]:
        """Extract cell percentages from text content."""
        cell_percentages = {}
        # Look for percentage patterns in the text
        ratio_pattern = r'(\w+)\s*Ratio:\s*["\']?([\d.]+)%["\']?'
        matches = re.finditer(ratio_pattern, text, re.IGNORECASE)
        
        for match in matches:
            cell_type = match.group(1).lower()
            value = match.group(2)
            if cell_type in ['neutrophils', 'eosinophils', 'lymphocytes', 'macrophages']:
                cell_percentages[cell_type] = f"{value}%"
        
        return cell_percentages

    async def __call__(
        self,
        question: Union[str, Dict[str, str]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Process a question through the RAG chain."""
        try:
            # Log input type and content
            logger.info(f"Chain input type: {type(question)}")
            logger.info(f"Chain input content: {question}")
            
            # Extract question and additional context if input is dictionary
            additional_context = ""
            image_analysis = None
            if isinstance(question, dict):
                additional_context = str(question.get("context", ""))
                image_analysis = question.get("image_analysis", None)
                question_text = str(question.get("question", ""))
            else:
                question_text = str(question)
            logger.info(f"Processing question: {question_text}")

            # Check if this is a cell count analysis request
            if "comment" in question_text.lower():
                try:
                    # First try to get percentages from image_analysis if available
                    cell_percentages = {}
                    if image_analysis and isinstance(image_analysis, dict):
                        for k, v in image_analysis.items():
                            if isinstance(v, str):
                                if '%' in v:
                                    cell_percentages[k] = v.strip('" ')
                                else:
                                    try:
                                        cell_percentages[k] = f"{float(v)*100:.1f}%"
                                    except ValueError:
                                        cell_percentages[k] = v
                            elif isinstance(v, (int, float)):
                                cell_percentages[k] = f"{float(v)*100:.1f}%"
                    
                    # If no percentages from image_analysis, try to extract from question text
                    if not cell_percentages:
                        cell_percentages = self._extract_cell_percentages_from_text(question_text)
                    
                    if cell_percentages:
                        # Generate analysis of the results
                        analysis = self._analyze_cell_counts(cell_percentages)
                        
                        # Format the response
                        response = (
                            "Cell Count Analysis:\n"
                            f"- Neutrophils: {cell_percentages.get('neutrophils', 'N/A')}\n"
                            f"- Eosinophils: {cell_percentages.get('eosinophils', 'N/A')}\n"
                            f"- Lymphocytes: {cell_percentages.get('lymphocytes', 'N/A')}\n"
                            f"- Macrophages: {cell_percentages.get('macrophages', 'N/A')}\n\n"
                            "Analysis:\n"
                            f"{analysis}\n\n"
                            "Normal BAL fluid contains:\n"
                            "- 80-90% alveolar macrophages\n"
                            "- 5-15% lymphocytes\n"
                            "- 1-3% neutrophils\n"
                            "- ≤1% eosinophils\n"
                        )
                        
                        logger.info(f"Generated cell count analysis response: {response}")
                        return {
                            "answer": response,
                            "source_documents": []
                        }
                
                except Exception as e:
                    logger.error(f"Error analyzing cell counts: {e}")
                    # Continue with normal processing if cell count analysis fails
            
            # If we get here, process as normal question
            # Get relevant documents
            logger.info(f"\nRetrieving documents for question: '{question_text}'")
            try:
                # Pass question directly to retriever
                retriever_response = await self.retriever.aget_relevant_documents(question_text)
                logger.info(f"Retrieved {len(retriever_response)} documents")
                
                if not retriever_response:
                    logger.warning("No relevant documents found!")
                    docs = []
                else:
                    docs = retriever_response
                    logger.info("\nRetrieved documents:")
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        logger.info(f"\n{i}. From {source}:")
                        logger.info(f"   Content preview: {doc.page_content[:200]}...")
            except Exception as e:
                logger.error(f"Retriever error: {str(e)}")
                docs = []
            
            # Format context from documents using string concatenation
            doc_texts = []
            logger.info("\nDocuments used for response (ordered by relevance):")
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                score = doc.metadata.get('score', 'N/A')
                logger.info(f"\n{i}. {source} (score: {score}):")
                logger.info(f"   First 100 chars: {doc.page_content[:100]}...")
                # Clean and format document content
                cleaned_content = self._clean_document_content(str(doc.page_content))
                doc_texts.append(cleaned_content)
            
            # Join and truncate context
            doc_context = "\n\n".join(doc_texts)
            doc_context = self._truncate_context(doc_context)
            
            # Build the full context
            context = doc_context
            if additional_context:
                context = additional_context + "\n\n" + context
            
            # Build prompt using string concatenation
            prompt = (
                self.prompt_prefix +
                context +
                self.prompt_question +
                question_text +
                self.prompt_suffix
            )
            
            logger.info(f"Final prompt type: {type(prompt)}")
            logger.info(f"Final prompt content: {prompt}")
            logger.info("Built prompt using string concatenation")
            
            # Get answer from LLM
            logger.info("About to call LLM with prompt")
            raw_answer = await self.llm._acall(prompt)
            logger.info(f"LLM response type: {type(raw_answer)}")
            logger.info(f"Raw LLM response: {raw_answer}")
            
            # Clean and format the response
            if isinstance(raw_answer, dict):
                answer = raw_answer.get("generated_text", "")
                if not answer:
                    answer = raw_answer.get("text", "")
                if not answer:
                    answer = raw_answer.get("answer", "")
                if not answer:
                    answer = str(raw_answer)
            else:
                answer = str(raw_answer)
            
            # Clean the response
            answer = self._clean_response(answer)
            
            # Return just the cleaned answer without references
            return {
                "answer": answer,
                "source_documents": docs  # Keep the source docs in return value but don't show in response
            }
            
        except Exception as e:
            logger.error("Error processing question through chain: " + str(e))
            raise 

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
                    "max_tokens": 4096,  # Increased from 2048 to allow longer responses
                    "top_p": 0.2,
                    "repeat_penalty": 2.0,
                }
                
                # Make the API call
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                answer = result['generated_text']
                
                return answer
            
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                last_error = e
                time.sleep(self.retry_delay)
        
        logger.error(f"All API calls failed after {self.max_retries} attempts")
        raise last_error 
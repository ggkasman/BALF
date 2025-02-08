"""
Main application module for the Meditron RAG system.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
import gradio as gr
import os
from pathlib import Path
import sys
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import tempfile

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to Python path
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Log initial paths
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Working directory: {os.getcwd()}")

# Import settings after setting up paths
from config.settings import (
    HOST,
    PORT,
    DEBUG,
    DOCS_DIR,
    MODELS_DIR,
    RESNET_MODEL_PATH,
    VECTOR_STORE_PATH
)

# Log configuration paths
logger.info(f"Model directory: {MODELS_DIR}")
logger.info(f"Model path: {RESNET_MODEL_PATH}")

from model.llm import MeditronLLM
from model.chain import MeditronChain
from model.image_analyzer import LeukocyteAnalyzer
from utils.document_loader import load_documents
from utils.embeddings import create_embeddings
from utils.vectorstore import initialize_vectorstore

class ChatMessage(BaseModel):
    """Schema for chat messages."""
    message: str
    history: List[List[str]] = Field(default_factory=list)

class ImageAnalysis(BaseModel):
    """Schema for image analysis results."""
    image_path: str
    results: Dict[str, float] = Field(default_factory=dict)

class MeditronApp(BaseModel):
    """Main application class."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=True,
        protected_namespaces=('model_', )
    )
    
    chain: Optional[MeditronChain] = None
    analyzer: Optional[LeukocyteAnalyzer] = None
    
    # Track current analysis state
    current_analysis: Dict[str, Any] = Field(default_factory=dict)
    has_analysis: bool = False
    
    async def initialize(self):
        """Initialize RAG components and image analyzer."""
        try:
            # Create LLM
            llm = MeditronLLM()
            logger.info("Created Meditron LLM")
            
            # Create embeddings
            embeddings = create_embeddings()
            logger.info("Created embeddings model")
            
            # Load documents
            documents = load_documents(DOCS_DIR)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Initialize vector store
            vector_store = initialize_vectorstore(embeddings, documents, str(VECTOR_STORE_PATH))
            logger.info("Initialized vector store")
            
            # Create chain
            self.chain = MeditronChain(llm, vector_store)
            logger.info("Created Meditron chain")
            
            # Initialize image analyzer with correct model path
            self.analyzer = LeukocyteAnalyzer(RESNET_MODEL_PATH)
            logger.info("Initialized LeukocyteAnalyzer")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {str(e)}")
            raise
            
    async def process_query(
        self,
        message: str,
        history: List[List[str]]  # [[user_msg1, bot_msg1], [user_msg2, bot_msg2]]
    ) -> List[List[str]]:  # Must return updated history
        """Process a query through the RAG chain."""
        try:
            if not self.chain:
                return history + [[message, "System initializing..."]]
            
            logger.info(f"process_query input type: {type(message)}")
            logger.info(f"process_query input content: {message}")

            # Add analysis results to context if available
            if self.has_analysis and "image" in message.lower() or "analysis" in message.lower():
                analysis_lines = []
                for cell_type, ratio in self.current_analysis.items():
                    percentage = f"{float(ratio) * 100:.2f}%"
                    analysis_lines.append(f"{cell_type}: {percentage}")
                analysis_text = "Current BAL Fluid Analysis Results:\n" + "\n".join(analysis_lines)
                
                # Format into a single string with context
                query = f"{message}\n\nContext:\n{analysis_text}"
            else:
                query = str(message)

            logger.info(f"Final query type before chain: {type(query)}")
            logger.info(f"Final query content before chain: {query}")
            
            # Process through chain
            raw_answer = await self.chain(query)
            
            # Handle different response formats
            if isinstance(raw_answer, dict):
                # Extract the answer from the dictionary
                answer = raw_answer.get("answer", "No answer found")
            else:
                answer = str(raw_answer)
            
            # Use the original message for history to maintain context
            original_message = query_text if isinstance(message, dict) else str(message)
            return history + [[original_message, answer]]
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return history + [[str(message), "Error processing question"]]
            
    def analyze_image(self, image: str) -> Dict[str, Any]:
        """Analyze image using LeukocyteAnalyzer.
        
        Args:
            image: Path to the image file
            
        Returns:
            Dictionary with analysis results or error message
        """
        try:
            if not self.analyzer:
                return {"error": "Image analyzer not initialized"}
            
            # Analyze the image
            results = self.analyzer.analyze(image)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    async def process_image_and_query(
        self,
        image: str,
        query: str,
        history: List[List[str]]
    ) -> List[List[str]]:
        """Process an image analysis followed by a query about the analysis."""
        try:
            # First analyze the image
            analysis_results = self.analyze_image(image)
            
            if "error" in analysis_results:
                return history + [[f"Image analysis with query: {query}", 
                                 f"Error during image analysis: {analysis_results['error']}"]]
            
            # Store analysis results for context
            self.current_analysis = analysis_results
            self.has_analysis = True
            
            # Convert analysis results to a formatted string
            analysis_lines = []
            for cell_type, ratio in analysis_results.items():
                # Convert ratio to percentage with 2 decimal places
                percentage = f"{float(ratio) * 100:.2f}%"
                analysis_lines.append(f"{cell_type}: {percentage}")
            analysis_text = "BAL Fluid Analysis Results:\n" + "\n".join(analysis_lines)
            
            # Add analysis results to history
            history = history + [["Image Analysis", analysis_text]]
            
            # Create query string (not a dictionary)
            query_with_context = (
                query + "\n\n" +
                "Context:\n" +
                analysis_text
            )
            
            # Process the query with analysis context
            return await self.process_query(query_with_context, history)
            
        except Exception as e:
            logger.error(f"Error in process_image_and_query: {str(e)}")
            return history + [[f"Image analysis with query: {query}", 
                             f"Error during processing: {str(e)}"]]

    async def process_combined_input(
        self,
        message: str,
        image: Optional[str],
        history: List[List[str]]
    ) -> List[List[str]]:
        """Process either an image analysis, a query, or both.
        
        Args:
            message: User's message/query
            image: Optional path to image file
            history: Chat history
            
        Returns:
            Updated chat history with response
        """
        try:
            # Case 1: New image analysis with query
            if image is not None:
                return await self.process_image_and_query(image, message, history)
                
            # Case 2: Query about existing analysis
            elif self.has_analysis:
                return await self.process_query(message, history)
                
            # Case 3: Query without any image context
            else:
                return await self.process_query(message, history)
                
        except Exception as e:
            logger.error(f"Error in process_combined_input: {str(e)}")
            return history + [[message, f"Error during processing: {str(e)}"]]

    async def handle_chat_message(
        self,
        message: str,
        history: List[List[str]]
    ) -> List[List[str]]:
        """Handle incoming chat messages.
        
        This function processes both regular medical queries and queries about image analysis.
        
        Args:
            message: The user's message/query
            history: Current chat history
            
        Returns:
            Updated chat history with response
        """
        try:
            # Process the message using combined input handler
            # Pass None as image since we're handling a chat message
            return await self.process_combined_input(message, None, history)
            
        except Exception as e:
            logger.error(f"Error handling chat message: {str(e)}")
            return history + [[message, f"Error processing message: {str(e)}"]]

    async def handle_image_analysis(
        self,
        image: str,
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle image analysis requests.
        
        Args:
            image: Path to the uploaded image
            current_results: Current analysis results displayed in the UI
            
        Returns:
            Updated analysis results
        """
        try:
            if not image:
                return {"error": "No image provided"}
                
            # Analyze the image (synchronous call)
            results = self.analyze_image(image)
            
            # Update application state
            if "error" not in results:
                self.current_analysis = results
                self.has_analysis = True
            else:
                self.has_analysis = False
                
            return results
            
        except Exception as e:
            logger.error(f"Error handling image analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

def create_interface(app: MeditronApp) -> gr.Blocks:
    """Create the Gradio interface with both image analysis and chat."""
    
    with gr.Blocks(
        title="Medical Analysis System",
        theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# Medical Analysis System")
        
        with gr.Tabs():
            # Image Analysis Tab
            with gr.Tab("Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload BAL Image",
                            source="upload",
                            tool="editor",
                            elem_id="image_upload",
                            interactive=True,
                            height=300,
                            width=400,
                            label_position="top",
                            text="Click to Upload or Drag and Drop"
                        )
                        analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                    with gr.Column():
                        results = gr.JSON(
                            label="Analysis Results",
                            show_label=True
                        )
                
                analyze_btn.click(
                    fn=app.handle_image_analysis,
                    inputs=[image_input, results],
                    outputs=results,
                    api_name="analyze_image"
                )
            
            # Chat Tab
            with gr.Tab("Medical Chat"):
                chatbot = gr.Chatbot()
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a medical question",
                        placeholder="Type your question here...",
                        show_label=True,
                        lines=2
                    )
                    send_btn = gr.Button("Send", variant="primary")
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                
                # Set up event handlers
                msg.submit(
                    fn=app.handle_chat_message,
                    inputs=[msg, chatbot],
                    outputs=chatbot,
                    api_name="chat"
                )
                
                send_btn.click(
                    fn=app.handle_chat_message,
                    inputs=[msg, chatbot],
                    outputs=chatbot,
                    api_name="chat"
                )
                
                clear.click(
                    fn=lambda: None,
                    inputs=None,
                    outputs=chatbot,
                    api_name="clear_chat"
                )
        
        gr.Markdown("""
        ## Usage Instructions
        
        ### Image Analysis
        1. Upload a BAL image
        2. Click 'Analyze Image' to get leukocyte ratios
        
        ### Medical Chat
        - Ask questions about BAL procedures and findings
        - Get evidence-based answers from medical literature
        """)
    
    return interface

async def main():
    """Main application entry point."""
    try:
        # Initialize application
        app = MeditronApp()
        await app.initialize()
        
        # Create interface
        interface = create_interface(app)
        
        # Launch the interface with FastAPI integration
        interface.queue()
        interface.launch(
            server_name=HOST,
            server_port=PORT,
            share=True,  # Enable sharing to generate a public URL
            favicon_path=None
        )
        
        # Keep the server running
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
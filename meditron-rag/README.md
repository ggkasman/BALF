# Meditron RAG

A Retrieval-Augmented Generation (RAG) system with integrated image analysis for medical applications, specifically focused on bronchoalveolar lavage (BAL) fluid analysis and related procedures.

## Features

- Medical question answering using a Hugging Face-hosted Meditron model
- BAL fluid image analysis using a ResNet50-based deep learning model
- Efficient document retrieval using FAISS vector store
- Interactive web interface with both chat and image analysis capabilities
- Asynchronous processing for better performance
- Configurable through environment variables
- Clean and modular architecture

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd meditron-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add:
- Your Hugging Face API token
- The Meditron endpoint URL
- Other configuration options

5. Add your medical documents:
Place your PDF documents in the `data/docs` directory.

6. Ensure the ResNet model is available:
Place the ResNet50 model file (`resnet50-cp-0171-0.0967.hdf5`) in the `src/model` directory.

## Usage

1. Start the application:
```bash
python src/app.py
```

2. Access the web interface:
Open your browser and navigate to `http://localhost:7860` (default port)

The interface provides two main features:
- **Image Analysis**: Upload BAL fluid images for automated cell count analysis
- **Medical Chat**: Ask questions about BAL procedures and get evidence-based answers


## Project Structure

```
meditron-rag/
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── model/
│   │   ├── llm.py
│   │   ├── chain.py
│   │   ├── image_analyzer.py
│   ├── utils/
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   └── app.py
├── data/
│   ├── docs/
│   └── vector_store/
├── images 
├── requirements.txt
└── README.md
```

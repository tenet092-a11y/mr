# SIH - Multimodal RAG Backend

A comprehensive multimodal Retrieval-Augmented Generation (RAG) system that processes text, images, and audio content for intelligent document search and question answering.

## Features

- **Multimodal Content Processing**: Supports text documents (PDF, DOCX), images (PNG, JPG, etc.), and audio files (MP3, WAV, etc.)
- **Unified Embedding Generation**: Cross-modal embeddings using sentence-transformers, CLIP, and Whisper
- **Intelligent Chunking**: Context-aware content segmentation with precise citation mapping
- **Vector Search**: Efficient similarity search using FAISS indexing
- **Local LLM Integration**: Privacy-focused response generation with citation support
- **REST API**: FastAPI-based interface for easy integration

## Architecture

The system consists of several key components:

- **Document Processors**: Handle different file formats and extract content
- **Embedding Generators**: Create unified vector representations for all content types
- **Vector Storage**: Efficient indexing and retrieval using FAISS
- **LLM Integration**: Local language model for response generation
- **API Layer**: RESTful interface for client applications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harikrishna-au/sih.git
cd sih
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models (optional - will be downloaded automatically on first use):
```bash
# Text embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# CLIP model for images
python -c "import clip; clip.load('ViT-B/32')"
```

## Usage

### Command Line Interface

Process documents:
```bash
python -m src.cli process-file path/to/document.pdf
python -m src.cli process-batch path/to/documents/
```

Query the system:
```bash
python -m src.cli query "What is the main topic discussed in the documents?"
```

### API Server

Start the API server:
```bash
python -m src.api.main
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Configuration

The system can be configured through environment variables or configuration files. Key settings include:

- `CHUNK_SIZE`: Size of content chunks (default: 512)
- `EMBEDDING_DEVICE`: Device for embedding generation (auto, cpu, cuda)
- `TEXT_MODEL_NAME`: Sentence transformer model name
- `LLM_MODEL_PATH`: Path to local LLM model file

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with sentence-transformers for text embeddings
- Uses OpenAI's CLIP for image understanding
- Powered by Whisper for audio transcription
- Vector search provided by FAISS
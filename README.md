
# Skin CareBot

A Streamlit-based application that uses Retrieval-Augmented Generation (RAG) to answer questions about common skin problems based on a provided PDF document. The project processes a PDF, extracts text and images, performs OCR, and builds a FAISS vectorstore for retrieval, then serves a web interface for querying.

## Description

Skin CareBot allows users to ask questions about skin conditions, leveraging a PDF document (`Common Skin Problem.pdf`) as the knowledge base. The backend uses LangChain, FAISS, and HuggingFace embeddings, with Grok (via the Groq API) for natural language responses. Images extracted from the PDF are displayed when relevant to the query.

## Prerequisites

- **Python**: Version 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
- **Groq API Key**: Obtain from [Groq](https://console.groq.com/keys)
- **Tesseract OCR**: Install for image text extraction (required for `main.py`)
  - Windows: Download from [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
  - Linux/macOS: Install via `apt-get install tesseract-ocr` or `brew install tesseract`
- **Poppler**: Required for PDF processing
  - Windows: Install via [poppler](https://github.com/oschwartz10612/poppler-windows) and add to PATH
  - Linux/macOS: Install via `apt-get install poppler-utils` or `brew install poppler`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

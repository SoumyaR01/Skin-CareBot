# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# import streamlit as st
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Load environment variables
# load_dotenv()

# # Langsmith Tracking
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = "true"
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# # Prompt Template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the question asked."),
#         ("user", "Question: {question}")
#     ]
# )

# # Streamlit Framework
# st.title("CareBot")
# input_text = st.text_input("Describe your symptoms or ask a medical question.")

# # Groq Llama3-70B-8192 model
# llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser

# if input_text:
#     st.write(chain.invoke({"question": input_text}))



# import os
# import logging
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # --- Setup ---
# load_dotenv()
# logging.basicConfig(filename='D:\\Udemy\\Lang_Chain\\1.2-Ollama\\output\\processing.log',
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("PDFChunkLogger")

# input_file = r"D:\Udemy\Lang_Chain\1.2-Ollama\input\Common Skin Problem.pdf"
# output_dir = r"D:\Udemy\Lang_Chain\1.2-Ollama\output"
# faiss_index_path = os.path.join(output_dir, "faiss_index")
# chunk_file_path = os.path.join(output_dir, "chunks.csv")

# os.makedirs(output_dir, exist_ok=True)

# # --- Processing ---
# def process_pdf_to_chunks_and_faiss():
#     logger.info("Starting PDF processing.")
#     if not os.path.exists(input_file):
#         logger.error(f"PDF not found: {input_file}")
#         return

#     # Load PDF
#     loader = PyPDFLoader(input_file)
#     docs = loader.load()
#     logger.info(f"Loaded {len(docs)} pages from PDF.")

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(docs)
#     logger.info(f"Split into {len(chunks)} chunks.")

#     # Save chunks to CSV
#     chunk_texts = [doc.page_content for doc in chunks]
#     chunk_df = pd.DataFrame({'chunk_id': range(len(chunk_texts)), 'chunk_text': chunk_texts})
#     chunk_df.to_csv(chunk_file_path, index=False, encoding='utf-8')
#     logger.info(f"Saved chunk file to {chunk_file_path}")

#     # Embeddings (Fixed: Explicit model for 768 dims consistency)
#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5",
#         model_kwargs={'device': 'cpu'}  # 'cuda' for GPU
#     )
#     logger.info("Initialized HuggingFace embeddings.")

#     # Create FAISS vectorstore and save locally
#     vectorstore = FAISS.from_documents(chunks, embeddings)
#     vectorstore.save_local(faiss_index_path)
#     logger.info(f"FAISS index saved to {faiss_index_path} (dims: {len(embeddings.embed_query('test'))})")  # Debug dim

#     print(f"Chunks saved in: {chunk_file_path}")
#     print(f"FAISS index saved in: {faiss_index_path}")

# if __name__ == "__main__":
#     process_pdf_to_chunks_and_faiss()


# import os
# import logging
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader, PdfWriter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter

# # --- Setup ---
# load_dotenv()
# logging.basicConfig(
#     filename='D:\\Udemy\\Lang_Chain\\1.2-Ollama\\output\\processing.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("PDFChunkLogger")

# input_file = r"D:\Udemy\Lang_Chain\1.2-Ollama\input\Common Skin Problem.pdf"
# output_dir = r"D:\Udemy\Lang_Chain\1.2-Ollama\output"
# faiss_index_path = os.path.join(output_dir, "faiss_index")
# pdf_chunks_dir = os.path.join(output_dir, "pdf_chunks")

# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(pdf_chunks_dir, exist_ok=True)

# # --- Helper: Save a list of text chunks as separate PDFs ---
# def save_chunks_as_pdfs(chunks, output_folder):
#     for i, chunk in enumerate(chunks):
#         writer = PdfWriter()
#         # Convert chunk text into a PDF page
#         # Here we simply write the text as a single page per chunk
#         # You could use reportlab for more advanced formatting if needed

#         chunk_pdf_path = os.path.join(output_folder, f"chunk_{i+1}.pdf")
#         c = canvas.Canvas(chunk_pdf_path, pagesize=letter)
#         textobject = c.beginText(40, 750)
#         for line in chunk.page_content.split('\n'):
#             textobject.textLine(line)
#         c.drawText(textobject)
#         c.save()

# # --- Processing ---
# def process_pdf_to_chunks_and_faiss():
#     logger.info("Starting PDF processing.")
    
#     if not os.path.exists(input_file):
#         logger.error(f"PDF not found: {input_file}")
#         return

#     # Load PDF
#     loader = PyPDFLoader(input_file)
#     docs = loader.load()
#     logger.info(f"Loaded {len(docs)} pages from PDF.")

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(docs)
#     logger.info(f"Split into {len(chunks)} chunks.")

#     # Save chunks as separate PDF files
#     save_chunks_as_pdfs(chunks, pdf_chunks_dir)
#     logger.info(f"Saved {len(chunks)} PDF chunks in {pdf_chunks_dir}")

#     # Embeddings
#     embeddings = HuggingFaceEmbeddings(
#         #model_name="BAAI/bge-base-en-v1.5",
#         model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#         model_kwargs={'device': 'cpu'}
#     )
#     logger.info("Initialized HuggingFace embeddings.")

#     # Create FAISS vectorstore
#     vectorstore = FAISS.from_documents(chunks, embeddings)
#     vectorstore.save_local(faiss_index_path)
#     logger.info(f"FAISS index saved to {faiss_index_path}")

#     print(f"PDF chunks saved in: {pdf_chunks_dir}")
#     print(f"FAISS index saved in: {faiss_index_path}")

# if __name__ == "__main__":
#     process_pdf_to_chunks_and_faiss()


import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfWriter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain_core.documents import Document
import fitz  # PyMuPDF for image extraction
from PIL import Image
import pytesseract
import io  

# --- Setup ---
load_dotenv()
logging.basicConfig(
    filename='D:\\Udemy\\Lang_Chain\\1.2-Ollama\\output\\processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDFChunkLogger")

input_file = r"D:\Udemy\Lang_Chain\1.2-Ollama\input\Common Skin Problem.pdf"
output_dir = r"D:\Udemy\Lang_Chain\1.2-Ollama\output"
faiss_index_path = os.path.join(output_dir, "faiss_index")
pdf_chunks_dir = os.path.join(output_dir, "pdf_chunks")
images_dir = os.path.join(output_dir, "images")
ocr_dir = os.path.join(output_dir, "ocr_texts")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(pdf_chunks_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(ocr_dir, exist_ok=True)

# --- Helper: Save a list of text chunks as separate PDFs ---
def save_chunks_as_pdfs(chunks, output_folder):
    for i, chunk in enumerate(chunks):
        chunk_pdf_path = os.path.join(output_folder, f"chunk_{i+1}.pdf")
        c = canvas.Canvas(chunk_pdf_path, pagesize=letter)
        textobject = c.beginText(40, 750)
        for line in chunk.page_content.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()
        logger.info(f"Saved PDF chunk {i+1} to {chunk_pdf_path}")

# --- Helper: Extract images and OCR from PDF ---
def extract_images_and_ocr(pdf_path):
    extracted = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            # Extract image
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha < 4:  # Skip transparency
                img_data = pix.tobytes("png")
                img_path = os.path.join(images_dir, f"img_page_{page_num:03d}_{img_index}.png")
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                # OCR on image
                image = Image.open(io.BytesIO(img_data))
                ocr_text = pytesseract.image_to_string(image)
                ocr_path = os.path.join(ocr_dir, f"ocr_page_{page_num:03d}_{img_index}.txt")
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
                
                extracted.append({
                    'image_path': img_path,
                    'ocr_text': ocr_text,
                    'page': page_num,
                    'position': img[1:]  # Bounding box for context
                })
                logger.info(f"Extracted image {img_index} from page {page_num} with OCR text length {len(ocr_text)}")
    
    doc.close()
    return extracted

# --- Processing ---
def process_pdf_to_chunks_and_faiss():
    logger.info("Starting PDF processing.")
    
    if not os.path.exists(input_file):
        logger.error(f"PDF not found: {input_file}")
        return

    # Extract images and OCR first
    all_images = extract_images_and_ocr(input_file)
    logger.info(f"Extracted {len(all_images)} images with OCR.")

    # Load PDF for text
    loader = PyPDFLoader(input_file)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDF.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} text chunks.")

    # Enrich chunks with OCR text and image metadata
    enriched_chunks = []
    page_ocr_map = {}  # page_num -> list of OCR texts
    page_img_map = {}  # page_num -> list of image paths
    for img in all_images:
        page = img['page']
        if page not in page_ocr_map:
            page_ocr_map[page] = []
            page_img_map[page] = []
        page_ocr_map[page].append(img['ocr_text'])
        page_img_map[page].append(img['image_path'])

    for chunk in chunks:
        page = chunk.metadata.get('page', 0)
        # Append OCR text to chunk content if available
        ocr_texts = page_ocr_map.get(page, [])
        chunk.page_content += "\n[OCR from images: " + " ".join(ocr_texts) + "]"
        # Add image paths to metadata
        chunk.metadata['images'] = page_img_map.get(page, [])
        enriched_chunks.append(chunk)
    
    logger.info(f"Enriched {len(enriched_chunks)} chunks with OCR and image metadata.")

    # Save enriched text chunks as separate PDF files
    save_chunks_as_pdfs(enriched_chunks, pdf_chunks_dir)
    logger.info(f"Saved {len(enriched_chunks)} enriched PDF chunks in {pdf_chunks_dir}")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    logger.info("Initialized HuggingFace embeddings.")

    # Create FAISS vectorstore with enriched chunks
    vectorstore = FAISS.from_documents(enriched_chunks, embeddings)
    vectorstore.save_local(faiss_index_path)
    logger.info(f"FAISS index saved to {faiss_index_path}")

    print(f"PDF chunks saved in: {pdf_chunks_dir}")
    print(f"FAISS index saved in: {faiss_index_path}")
    print(f"Extracted {len(all_images)} images with OCR.")

if __name__ == "__main__":
    process_pdf_to_chunks_and_faiss()
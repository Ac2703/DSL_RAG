from modules.webscraping import WebScraper

import re
import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from langchain.schema import Document

import nltk
from nltk.tokenize import sent_tokenize

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


nltk.download("punkt")


# Initialize Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)


class Chunker:
    def __init__(self, course_dict, chunk_size=500, chunk_overlap=50):
        self.course_dict = course_dict
        self.url = course_dict.get("syllabus", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def download_pdf(self, save_path="./temp.pdf"):
        try:
            response = requests.get(self.url, verify=False, timeout=15)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path
        except Exception as e:
            print(f"Failed to download PDF: {e}")
            return None
        
    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # Attach metadata safely
        for d in docs:
            d.metadata["source"] = self.url
            d.metadata["course_no"] = self.course_dict.get("course_num", "")
            d.metadata["course_sem"] = self.course_dict.get("course_sem", "")
            d.metadata["course_name"] = self.course_dict.get("course_name", "")
        return docs
    
    def load_html(self):
        try:
            response = requests.get(self.url, verify=False, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract visible text from the HTML body
            raw_text = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
            clean_text = re.sub(r'\s+', ' ', raw_text)
            
            # Create a single Document with metadata
            doc = Document(page_content=clean_text, metadata={
                "source": self.url,
                "course_no": self.course_dict.get("course_num", ""),
                "course_sem": self.course_dict.get("course_sem", ""),
                "course_name": self.course_dict.get("course_name", ""),
            })
            return [doc]  # Return list of docs for consistency

        except Exception as e:
            print(f"Failed to load HTML: {e}")
            return []
    
    def chunk_documents(self, docs):
        """
        This method:
         - For each Document, splits its text into sentence-based chunks up to chunk_size,
         - Ensures no sentences are cut in half,
         - Wraps each chunk back into a Document with metadata.
        """
        all_chunks = []

        for doc in docs:
            sentences = sent_tokenize(doc.page_content)  # split text into sentences
            current_chunk = ""

            for sentence in sentences:
                addition = (" " + sentence) if current_chunk else sentence
                if len(current_chunk) + len(addition) <= self.chunk_size:
                    current_chunk += addition
                else:
                    # Save the full chunk and start a new one
                    all_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata.copy()))
                    current_chunk = sentence

            if current_chunk:
                all_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata.copy()))

        return all_chunks

    def save_chunks_to_csv(self, chunks, filename="output.csv"):
        rows = []
        for chunk in chunks:
            rows.append({
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "course_no": chunk.metadata.get("course_no", ""),
                "course_sem": chunk.metadata.get("course_sem", ""),
                "course_name": chunk.metadata.get("course_name", ""),
            })
        df = pd.DataFrame(rows)


        # Append mode with header only if file doesn't exist
        write_header = not os.path.exists(filename)

        df.to_csv(filename, index=False, mode='a', header=write_header)
        
        print(f"Saved {len(chunks)} chunks to {filename}")

    def process(self):
        if self.url.lower().endswith(".pdf"):
            pdf_path = self.download_pdf()
            if not pdf_path:
                print("PDF download failed, aborting.")
                return []
            docs = self.load_pdf(pdf_path)
            os.remove(pdf_path)  # clean up temp file
        else:
            docs = self.load_html()

        if not docs:
            print("No documents loaded, aborting.")
            return []

        chunks = self.chunk_documents(docs)
        print(f"Created {len(chunks)} chunks")
        return chunks


    
if __name__ == "__main__":
    pass
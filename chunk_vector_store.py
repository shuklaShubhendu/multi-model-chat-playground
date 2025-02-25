
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from langchain_openai import OpenAIEmbeddings

import re

class LegalChunkVectorStore:
    def __init__(self) -> None:
        self.embedding = OpenAIEmbeddings()
        
        # Define legal-specific chunk parameters
        self.chunk_params = {
            "Contract": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", ".", ";"]
            },
            "Legal Brief": {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", ". ", "; "]
            },
            "Court Document": {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", ". ", "; "]
            },
            "Legislation": {
                "chunk_size": 768,
                "chunk_overlap": 75,
                "separators": ["\n\n", "\n", ". ", "; "]
            },
            "Regulatory Filing": {
                "chunk_size": 768,
                "chunk_overlap": 75,
                "separators": ["\n\n", "\n", ". ", "; "]
            },
            "Other": {
                "chunk_size": 1024,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", ". ", "; "]
            },
            "Default": {
                "chunk_size": 2000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", ". ", "; "]
            }
        }

    def preprocess_legal_text(self, text):
        """Preprocess legal text to improve chunking quality"""
        # Standardize section markers
        text = re.sub(r'Section\s+(\d+)', r'§\1', text, flags=re.IGNORECASE)
        
        # Preserve list formatting
        text = re.sub(r'(\([a-z]\)|\d+\.)', r'\n\1', text)
        
        # Standardize paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Preserve legal citations
        text = re.sub(r'(\d+\s+[A-Za-z\.]+\s+\d+)', r' \1 ', text)
        
        return text

    def extract_legal_metadata(self, doc):
        """Extract legal-specific metadata from the document"""
        metadata = {}
        
        # Extract potential section headers
        section_headers = re.findall(r'^[A-Z][^.!?]*[:.]', doc, re.MULTILINE)
        if section_headers:
            metadata['section_headers'] = section_headers

        # Extract potential defined terms
        defined_terms = re.findall(r'"([^"]+)"\s+means', doc)
        if defined_terms:
            metadata['defined_terms'] = defined_terms

        # Extract potential references
        references = re.findall(r'(?:Section|§)\s+\d+(?:\.\d+)*', doc)
        if references:
            metadata['references'] = references

        return metadata
    

    def split_into_chunks(self, file_path: str, document_type: str ):
        """Split legal documents into chunks with type-specific parameters"""
        # Load document
        
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == "docx":
            doc = Docx2txtLoader(file_path).load()
        elif file_extension == "pdf":
            doc = PyPDFLoader(file_path).load()
        elif file_extension == "txt":
            doc = TextLoader(file_path).load()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Get chunk parameters for document type
        params = self.chunk_params.get(document_type, self.chunk_params["Other"])
        
        # Create text splitter with document-type specific parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=params["separators"],
            length_function=len,
            add_start_index=True,
        )
        
        # Preprocess and split documents
        processed_docs = []
        for page in doc:
            # Preprocess the text
            processed_text = self.preprocess_legal_text(page.page_content)
            
            # Extract legal metadata
            legal_metadata = self.extract_legal_metadata(processed_text)
            
            # Update page content and metadata
            page.page_content = processed_text
            page.metadata.update(legal_metadata)
            processed_docs.append(page)

        # Split into chunks
        chunks = text_splitter.split_documents(processed_docs)
        chunks = filter_complex_metadata(chunks)
        
        return chunks
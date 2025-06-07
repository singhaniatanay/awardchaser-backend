#!/usr/bin/env python3
"""
Document Ingestion CLI Script

Downloads PDFs and TXT files from S3, processes them into chunks, and stores in Qdrant vector database.
Usage: python -m app.services.ingest --s3-prefix credit_docs/
"""

import argparse
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional
import sys
from io import BytesIO

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.config import settings
from app.core.retrieval import get_vector_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles document ingestion from S3 to Qdrant vector database."""
    
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt']
    
    def __init__(self, s3_bucket: Optional[str] = None):
        """
        Initialize document ingestor.
        
        Args:
            s3_bucket: S3 bucket name. Uses settings.s3_bucket if not provided.
        """
        self.s3_bucket = s3_bucket or settings.s3_bucket
        self.s3_client = boto3.client('s3',aws_access_key_id=settings.aws_key,aws_secret_access_key=settings.aws_secret)
        self.vector_client = get_vector_client()
        
        # Initialize text splitter with reasonable defaults for document content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Initialized DocumentIngestor for bucket: {self.s3_bucket}")
        logger.info(f"Supported file types: {', '.join(self.SUPPORTED_EXTENSIONS)}")
    
    def list_s3_documents(self, prefix: str) -> List[str]:
        """
        List supported document files in S3 bucket with given prefix.
        
        Args:
            prefix: S3 key prefix to filter objects
            
        Returns:
            List of S3 keys for supported document files
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            document_keys = []
            
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Check if file has a supported extension
                        if any(key.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                            document_keys.append(key)
            
            logger.info(f"Found {len(document_keys)} supported document files with prefix '{prefix}'")
            return document_keys
            
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your credentials.")
            raise
    
    def download_document_from_s3(self, s3_key: str) -> bytes:
        """
        Download document content from S3.
        
        Args:
            s3_key: S3 key for the document file
            
        Returns:
            Document content as bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            document_content = response['Body'].read()
            logger.debug(f"Downloaded document: {s3_key} ({len(document_content)} bytes)")
            return document_content
            
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_content: bytes, source_key: str) -> str:
        """
        Extract text content from PDF bytes.
        
        Args:
            pdf_content: PDF file content as bytes
            source_key: S3 key of the source PDF for logging
            
        Returns:
            Extracted text content
        """
        try:
            pdf_stream = BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_stream)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {source_key}: {e}")
                    continue
            
            logger.debug(f"Extracted {len(text_content)} characters from PDF {source_key}")
            return text_content
            
        except Exception as e:
            logger.error(f"Error processing PDF {source_key}: {e}")
            raise
    
    def extract_text_from_txt(self, txt_content: bytes, source_key: str) -> str:
        """
        Extract text content from TXT bytes.
        
        Args:
            txt_content: TXT file content as bytes
            source_key: S3 key of the source TXT for logging
            
        Returns:
            Extracted text content
        """
        try:
            # Try to decode with common encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text_content = txt_content.decode(encoding)
                    logger.debug(f"Successfully decoded {source_key} using {encoding} encoding")
                    logger.debug(f"Extracted {len(text_content)} characters from TXT {source_key}")
                    return text_content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            text_content = txt_content.decode('utf-8', errors='replace')
            logger.warning(f"Used fallback decoding for {source_key}, some characters may be corrupted")
            logger.debug(f"Extracted {len(text_content)} characters from TXT {source_key}")
            return text_content
            
        except Exception as e:
            logger.error(f"Error processing TXT {source_key}: {e}")
            raise
    
    def get_file_type(self, s3_key: str) -> str:
        """
        Determine file type from S3 key extension.
        
        Args:
            s3_key: S3 key for the file
            
        Returns:
            File type ('pdf' or 'txt')
        """
        key_lower = s3_key.lower()
        if key_lower.endswith('.pdf'):
            return 'pdf'
        elif key_lower.endswith('.txt'):
            return 'txt'
        else:
            raise ValueError(f"Unsupported file type for {s3_key}")
    
    def extract_text_from_document(self, document_content: bytes, source_key: str) -> str:
        """
        Extract text content from document based on file type.
        
        Args:
            document_content: Document file content as bytes
            source_key: S3 key of the source document for logging and type detection
            
        Returns:
            Extracted text content
        """
        file_type = self.get_file_type(source_key)
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(document_content, source_key)
        elif file_type == 'txt':
            return self.extract_text_from_txt(document_content, source_key)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def create_document_chunks(self, text: str, metadata: dict) -> List[Document]:
        """
        Split text into chunks and create Document objects.
        
        Args:
            text: Text content to split
            metadata: Metadata to attach to each document chunk
            
        Returns:
            List of Document objects
        """
        try:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.debug(f"Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error creating document chunks: {e}")
            raise
    
    def process_document(self, s3_key: str) -> int:
        """
        Process a single document: download, extract text, chunk, and store in vector DB.
        
        Args:
            s3_key: S3 key for the document file
            
        Returns:
            Number of document chunks created and stored
        """
        try:
            file_type = self.get_file_type(s3_key)
            logger.info(f"Processing {file_type.upper()} document: {s3_key}")
            
            # Download document from S3
            document_content = self.download_document_from_s3(s3_key)
            
            # Extract text from document
            text_content = self.extract_text_from_document(document_content, s3_key)
            
            if not text_content.strip():
                logger.warning(f"No text extracted from {s3_key}, skipping")
                return 0
            
            # Create metadata
            metadata = {
                "source": s3_key,
                "source_type": f"s3_{file_type}",
                "bucket": self.s3_bucket,
                "file_type": file_type
            }
            
            # Create document chunks
            documents = self.create_document_chunks(text_content, metadata)
            
            if not documents:
                logger.warning(f"No document chunks created for {s3_key}")
                return 0
            
            # Store in vector database
            self.vector_client.add_documents(documents)
            
            logger.info(f"Successfully processed {s3_key}: {len(documents)} chunks stored")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to process document {s3_key}: {e}")
            raise
    
    def ingest_documents(self, s3_prefix: str) -> dict:
        """
        Ingest all supported documents from S3 prefix into vector database.
        
        Args:
            s3_prefix: S3 prefix to search for document files
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            logger.info(f"Starting document ingestion from s3://{self.s3_bucket}/{s3_prefix}")
            
            # Get list of documents
            document_keys = self.list_s3_documents(s3_prefix)
            
            if not document_keys:
                logger.warning(f"No supported document files found with prefix '{s3_prefix}'")
                return {"total_files": 0, "processed_files": 0, "total_chunks": 0, "failed_files": 0, "file_types": {}}
            
            # Track statistics by file type
            file_type_stats = {}
            total_chunks = 0
            processed_files = 0
            failed_files = 0
            
            for document_key in document_keys:
                try:
                    file_type = self.get_file_type(document_key)
                    chunks_created = self.process_document(document_key)
                    
                    # Update file type statistics
                    if file_type not in file_type_stats:
                        file_type_stats[file_type] = {"count": 0, "chunks": 0}
                    file_type_stats[file_type]["count"] += 1
                    file_type_stats[file_type]["chunks"] += chunks_created
                    
                    total_chunks += chunks_created
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {document_key}: {e}")
                    failed_files += 1
                    continue
            
            # Summary
            stats = {
                "total_files": len(document_keys),
                "processed_files": processed_files,
                "failed_files": failed_files,
                "total_chunks": total_chunks,
                "file_types": file_type_stats
            }
            
            logger.info(f"Ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents (PDF, TXT) from S3 into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.services.ingest --s3-prefix credit_docs/
  python -m app.services.ingest --s3-prefix documents/manuals/ --s3-bucket my-bucket
  
Supported file types: PDF, TXT
        """
    )
    
    parser.add_argument(
        '--s3-prefix',
        required=True,
        help='S3 prefix to search for document files (e.g., credit_docs/)'
    )
    
    parser.add_argument(
        '--s3-bucket',
        help='S3 bucket name (defaults to config setting)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Text chunk size for splitting (default: 1000)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Text chunk overlap size (default: 200)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Validate configuration
        if not settings.qdrant_url:
            logger.error("QDRANT_URL not configured. Please set environment variable.")
            sys.exit(1)
        
        if not settings.openai_api_key:
            logger.error("OPENAI_API_KEY not configured. Please set environment variable.")
            sys.exit(1)
        
        if not args.s3_bucket and not settings.s3_bucket:
            logger.error("S3 bucket not specified. Use --s3-bucket or set S3_BUCKET environment variable.")
            sys.exit(1)
        
        # Initialize ingestor
        ingestor = DocumentIngestor(s3_bucket=args.s3_bucket)
        
        # Update text splitter settings if provided
        if args.chunk_size != 1000 or args.chunk_overlap != 200:
            ingestor.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info(f"Updated text splitter: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
        
        # Run ingestion
        stats = ingestor.ingest_documents(args.s3_prefix)
        
        # Print results
        print("\n" + "="*50)
        print("INGESTION COMPLETE")
        print("="*50)
        print(f"Total files found: {stats['total_files']}")
        print(f"Successfully processed: {stats['processed_files']}")
        print(f"Failed files: {stats['failed_files']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        
        # Print file type breakdown
        if stats['file_types']:
            print("\nFile Type Breakdown:")
            print("-" * 30)
            for file_type, type_stats in stats['file_types'].items():
                print(f"{file_type.upper()}: {type_stats['count']} files, {type_stats['chunks']} chunks")
        
        print("="*50)
        
        if stats['failed_files'] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
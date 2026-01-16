"""
Cloud-based document processor using API embeddings.
Uses Together.ai for embeddings (very cheap: $0.008/1M tokens).
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import hashlib
import os
from openai import OpenAI

from config.settings import settings


class DocumentProcessorCloud:
    """Processor for chunking documents and creating embeddings via API."""

    def __init__(self):
        """Initialize the document processor with cloud embedding model."""

        # Together.ai client (OpenAI-compatible)
        self.embedding_client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY")
        )

        # Together.ai embedding model (very cheap)
        self.embedding_model_name = os.getenv(
            "CLOUD_EMBEDDING_MODEL",
            "togethercomputer/m2-bert-80M-8k-retrieval"
        )

        # Dimension depends on model:
        # m2-bert-80M-8k-retrieval: 768 dims
        # WhereIsAI/UAE-Large-V1: 1024 dims
        self.embedding_dimension = 768

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print(f"Cloud embedding model: {self.embedding_model_name}")
        print(f"Embedding dimension: {self.embedding_dimension}")
        print(f"Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents with 'text', 'title', and 'url'

        Returns:
            List of chunks with metadata
        """
        print("\nChunking documents...")
        chunks = []

        for doc in documents:
            # Split the content into chunks
            text_chunks = self.text_splitter.split_text(doc['text'])

            # Create chunk documents with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'content': chunk_text,
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
                chunks.append(chunk)

        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def create_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict]:
        """
        Create embeddings for all chunks using Together.ai API.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            List of chunks with embeddings
        """
        print("\nGenerating embeddings via API...")
        print(f"Processing {len(chunks)} chunks...")

        # Process in batches to avoid rate limits
        batch_size = 100

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]

            try:
                # Call Together.ai embeddings API
                response = self.embedding_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model_name
                )

                # Attach embeddings to chunks
                for chunk, embedding_data in zip(batch, response.data):
                    chunk['embedding'] = embedding_data.embedding

            except Exception as e:
                print(f"\nError generating embeddings for batch {i//batch_size}: {e}")
                raise

        return chunks

    def index_to_qdrant(self, chunks: List[Dict]):
        """
        Index chunks with embeddings into Qdrant.

        Args:
            chunks: List of chunks with embeddings and metadata
        """
        print("\nConnecting to Qdrant...")

        # Support both local and cloud Qdrant
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_api_key:
            # Qdrant Cloud
            client = QdrantClient(
                url=f"https://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                api_key=qdrant_api_key
            )
        else:
            # Local Qdrant
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )

        collection_name = settings.QDRANT_COLLECTION_NAME

        # Recreate collection (delete if exists)
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dimension,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")

        # Prepare points for insertion
        points = []
        for idx, chunk in enumerate(tqdm(chunks, desc="Preparing points")):
            point = PointStruct(
                id=idx,
                vector=chunk['embedding'],
                payload={
                    'content': chunk['content'],
                    'title': chunk['title'],
                    'url': chunk['url'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks']
                }
            )
            points.append(point)

        # Upload points in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"\n✓ Successfully indexed {len(points)} chunks to Qdrant!")

        # Display collection info
        collection_info = client.get_collection(collection_name)
        print(f"\nCollection Info:")
        print(f"  Name: {collection_info.name}")
        print(f"  Vectors count: {collection_info.vectors_count}")
        print(f"  Points count: {collection_info.points_count}")

    def process_and_index(self, documents: List[Dict[str, str]]):
        """
        Complete pipeline: chunk, embed, and index documents.

        Args:
            documents: List of scraped documents
        """
        if not documents:
            print("No documents to process!")
            return

        # Step 1: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 2: Create embeddings via API
        chunks_with_embeddings = self.create_embeddings(chunks)

        # Step 3: Index to Qdrant
        self.index_to_qdrant(chunks_with_embeddings)

        print("\n" + "=" * 50)
        print("Indexing complete!")
        print("=" * 50)

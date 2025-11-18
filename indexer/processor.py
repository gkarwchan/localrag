"""
Document processor for chunking and vectorization.
Uses LangChain for text splitting and sentence-transformers for local embeddings.
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import hashlib

from config.settings import settings


class DocumentProcessor:
    """Processor for chunking documents and creating embeddings."""

    def __init__(self):
        """Initialize the document processor with embedding model."""
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print(f"Embedding dimension: {self.embedding_dimension}")
        print(f"Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents with 'content', 'title', and 'url'

        Returns:
            List of chunks with metadata
        """
        print("\nChunking documents...")
        chunks = []

        for doc in documents:
            # Split the content into chunks
            text_chunks = self.text_splitter.split_text(doc['content'])

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
        Create embeddings for all chunks.

        Args:
            chunks: List of text chunks with metadata

        Returns:
            List of chunks with embeddings
        """
        print("\nGenerating embeddings...")

        # Extract text for embedding
        texts = [chunk['content'] for chunk in chunks]

        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()

        return chunks

    def index_to_qdrant(self, chunks: List[Dict]):
        """
        Index chunks with embeddings into Qdrant.

        Args:
            chunks: List of chunks with embeddings and metadata
        """
        print("\nConnecting to Qdrant...")
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
            # Create a unique ID for each chunk
            chunk_id = hashlib.md5(
                f"{chunk['url']}_{chunk['chunk_index']}".encode()
            ).hexdigest()

            point = PointStruct(
                id=idx,  # Use index as ID for simplicity
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

        # Step 2: Create embeddings
        chunks_with_embeddings = self.create_embeddings(chunks)

        # Step 3: Index to Qdrant
        self.index_to_qdrant(chunks_with_embeddings)

        print("\n" + "=" * 50)
        print("Indexing complete!")
        print("=" * 50)

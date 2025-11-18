"""
RAG (Retrieval-Augmented Generation) engine for querying the indexed documents.
Handles embedding queries, searching Qdrant, and generating answers with LLM.
"""
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

from config.settings import settings


class RAGEngine:
    """RAG engine for question answering over indexed documents."""

    def __init__(self, model_name: str = None):
        """
        Initialize the RAG engine.

        Args:
            model_name: Ollama model to use (e.g., 'llama3.2', 'mistral')
        """
        self.model_name = model_name or settings.DEFAULT_LLM_MODEL

        # Initialize embedding model (same as used for indexing)
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

        print(f"RAG Engine initialized with model: {self.model_name}")

    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a query.

        Args:
            query: User's question

        Returns:
            Query embedding vector
        """
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def search_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for relevant documents in Qdrant.

        Args:
            query: User's question
            top_k: Number of results to return

        Returns:
            List of relevant document chunks with metadata
        """
        top_k = top_k or settings.TOP_K_RESULTS

        # Embed the query
        query_vector = self.embed_query(query)

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

        # Format results
        results = []
        for hit in search_results:
            results.append({
                'content': hit.payload['content'],
                'title': hit.payload['title'],
                'url': hit.payload['url'],
                'score': hit.score,
                'chunk_index': hit.payload.get('chunk_index', 0)
            })

        return results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Generate an answer using the LLM based on retrieved context.

        Args:
            query: User's question
            context_docs: Retrieved relevant documents
            conversation_history: Previous conversation messages

        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc['title']} ({doc['url']})\n{doc['content']}"
            for doc in context_docs
        ])

        # Build the prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
Your answers should be:
- Accurate and based only on the given context
- Clear and concise
- Helpful and informative
- If the context doesn't contain enough information to answer the question, say so honestly

Always cite the sources when appropriate."""

        user_prompt = f"""Context from the documentation:

{context}

Question: {query}

Please provide a helpful answer based on the context above."""

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Keep last 3 exchanges

        messages.append({"role": "user", "content": user_prompt})

        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": settings.TEMPERATURE,
                    "num_predict": settings.MAX_TOKENS
                }
            )
            return response['message']['content']

        except Exception as e:
            return f"Error generating answer: {str(e)}\n\nPlease make sure Ollama is running and the model '{self.model_name}' is installed."

    def generate_followup_questions(
        self,
        query: str,
        answer: str,
        context_docs: List[Dict]
    ) -> List[str]:
        """
        Generate follow-up questions based on the query and answer.

        Args:
            query: Original user question
            answer: Generated answer
            context_docs: Retrieved context documents

        Returns:
            List of 3 follow-up questions
        """
        prompt = f"""Based on this question and answer, suggest 3 relevant follow-up questions that the user might want to ask next.

Original Question: {query}

Answer: {answer}

Generate 3 concise follow-up questions (one per line, without numbering):"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.8,
                    "num_predict": 150
                }
            )

            # Parse the response into individual questions
            questions = [
                q.strip().lstrip('0123456789.-) ')
                for q in response['response'].strip().split('\n')
                if q.strip()
            ]

            # Return up to 3 questions
            return questions[:3]

        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []

    def query(
        self,
        question: str,
        conversation_history: List[Dict] = None
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Complete RAG pipeline: search, generate answer, and create follow-ups.

        Args:
            question: User's question
            conversation_history: Previous conversation messages

        Returns:
            Tuple of (answer, source_documents, followup_questions)
        """
        # Step 1: Search for relevant documents
        context_docs = self.search_documents(question)

        # Step 2: Generate answer
        answer = self.generate_answer(question, context_docs, conversation_history)

        # Step 3: Generate follow-up questions
        followup_questions = self.generate_followup_questions(
            question,
            answer,
            context_docs
        )

        return answer, context_docs, followup_questions

    def switch_model(self, model_name: str):
        """
        Switch to a different Ollama model.

        Args:
            model_name: Name of the Ollama model to switch to
        """
        self.model_name = model_name
        print(f"Switched to model: {self.model_name}")

    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models.

        Returns:
            List of model names
        """
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            print(f"Error getting models: {e}")
            return [settings.DEFAULT_LLM_MODEL]

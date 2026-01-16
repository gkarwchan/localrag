"""
Cloud-based RAG engine using API services for both LLM and embeddings.
- LLM: Groq (free tier with Llama 3.2)
- Embeddings: Together.ai (very cheap)
"""
import os
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from openai import OpenAI

from config.settings import settings


class RAGEngineCloud:
    """RAG engine using cloud APIs for question answering."""

    def __init__(self, model_name: str = None):
        """
        Initialize the cloud RAG engine.

        Args:
            model_name: Groq model to use (default: llama-3.2-90b-text-preview)
        """
        # LLM: Groq (free tier)
        self.llm_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_name = model_name or os.getenv(
            "GROQ_MODEL",
            "llama-3.2-90b-text-preview"
        )

        # Embeddings: Together.ai
        self.embedding_client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY")
        )
        self.embedding_model = os.getenv(
            "CLOUD_EMBEDDING_MODEL",
            "togethercomputer/m2-bert-80M-8k-retrieval"
        )

        # Qdrant (cloud or local)
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_api_key:
            self.qdrant_client = QdrantClient(
                url=f"https://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                api_key=qdrant_api_key
            )
        else:
            self.qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )

        print(f"RAG Engine initialized (Cloud)")
        print(f"  LLM: {self.model_name} (Groq)")
        print(f"  Embeddings: {self.embedding_model} (Together.ai)")

    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a query using Together.ai API.

        Args:
            query: User's question

        Returns:
            Query embedding vector
        """
        response = self.embedding_client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        return response.data[0].embedding

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
        Generate an answer using Groq LLM based on retrieved context.

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

        # Generate response using Groq
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error generating answer: {str(e)}\n\nPlease check your GROQ_API_KEY is set correctly."

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
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )

            # Parse the response into individual questions
            questions = [
                q.strip().lstrip('0123456789.-) ')
                for q in response.choices[0].message.content.strip().split('\n')
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
        Switch to a different Groq model.

        Args:
            model_name: Name of the Groq model to switch to
        """
        self.model_name = model_name
        print(f"Switched to model: {self.model_name}")

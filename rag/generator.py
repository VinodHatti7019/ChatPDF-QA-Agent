"""Generator Module - LLM answer generation with citations"""

import logging
from typing import List, Dict, Optional

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using LLM with context from retrieved documents.
    
    Args:
        model_name: LLM model to use
        api_key: API key for LLM provider
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            openai.api_key = api_key
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """Generate answer based on query and context.
        
        Args:
            query: User query
            context_docs: Retrieved documents with text and metadata
            system_prompt: Optional system prompt
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Build context from retrieved documents
        context = self._build_context(context_docs)
        
        # Build system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. Answer the question based on the provided context. "
                "If the answer is not in the context, say so. Cite sources when relevant."
            )
        
        # Build user message
        user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        response = self._call_llm(system_prompt, user_message)
        
        return {
            'answer': response,
            'sources': [doc.get('metadata', {}) for doc in context_docs]
        }
    
    def _build_context(self, context_docs: List[Dict]) -> str:
        """Build context string from documents."""
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            
            context_parts.append(f"[{i}] Source: {source} (Page {page})\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call LLM API."""
        if openai is None:
            raise ImportError("OpenAI not installed")
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

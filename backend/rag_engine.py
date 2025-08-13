import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
import pickle
from pathlib import Path
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
import tiktoken
from abc import ABC, abstractmethod
import asyncio
import aiofiles
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    chunks: List['DocumentChunk'] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    
@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    id: str
    document_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_idx: int = 0
    end_idx: int = 0
    
    
@dataclass
class SearchResult:
    """Represents a search result"""
    chunk: DocumentChunk
    score: float
    document: Optional[Document] = None
    
    
@dataclass
class ConversationSummary:
    """Represents a conversation summary"""
    id: str
    conversation_id: str
    summary: str
    key_points: List[str]
    entities: List[Dict[str, str]]
    topics: List[str]
    sentiment: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


class TextChunker:
    """Splits text into overlapping chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into chunks with overlap"""
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        doc_id = hashlib.md5(text.encode()).hexdigest()
        
        # Create chunks with overlap
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            # Find actual character positions
            chunk_text = self.tokenizer.decode(chunk_tokens)
            start_idx = text.find(chunk_text[:50])  # Find approximate start
            end_idx = start_idx + len(chunk_text)
            
            chunk = DocumentChunk(
                id=f"{doc_id}_{i}",
                document_id=doc_id,
                content=chunk_text,
                metadata=metadata or {},
                start_idx=start_idx,
                end_idx=end_idx
            )
            chunks.append(chunk)
            
        return chunks
        
    def chunk_by_sentences(self, text: str, max_chunk_size: int = 512) -> List[DocumentChunk]:
        """Split text into chunks by sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        doc_id = hashlib.md5(text.encode()).hexdigest()
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_size + sentence_tokens > max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = DocumentChunk(
                    id=f"{doc_id}_{chunk_idx}",
                    document_id=doc_id,
                    content=chunk_text,
                    metadata={}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap (last sentence)
                current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                current_size = len(self.tokenizer.encode(current_chunk[0])) if current_chunk else 0
                chunk_idx += 1
                
            current_chunk.append(sentence)
            current_size += sentence_tokens
            
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = DocumentChunk(
                id=f"{doc_id}_{chunk_idx}",
                document_id=doc_id,
                content=chunk_text,
                metadata={}
            )
            chunks.append(chunk)
            
        return chunks


class EmbeddingModel:
    """Handles text embeddings for semantic search"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text"""
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
        
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        pass
        
    @abstractmethod
    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
        
    @abstractmethod
    async def delete_document(self, doc_id: str):
        """Delete a document"""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_model: EmbeddingModel, index_path: Optional[str] = None):
        self.embedding_model = embedding_model
        self.index_path = index_path or "faiss_index"
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        
        # Initialize or load index
        if os.path.exists(f"{self.index_path}.index"):
            self.index = faiss.read_index(f"{self.index_path}.index")
            self._load_metadata()
        else:
            self.index = faiss.IndexFlatL2(embedding_model.embedding_dim)
            
    async def add_documents(self, documents: List[Document]):
        """Add documents to the FAISS index"""
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            # Store document
            self.documents[doc.id] = doc
            
            # Process chunks
            for chunk in doc.chunks:
                if chunk.embedding is None:
                    chunk.embedding = self.embedding_model.embed_text(chunk.content)[0]
                    
                self.chunks[chunk.id] = chunk
                all_chunks.append(chunk)
                all_embeddings.append(chunk.embedding)
                
        # Add to FAISS index
        if all_embeddings:
            embeddings_array = np.array(all_embeddings).astype('float32')
            self.index.add(embeddings_array)
            
        # Save index and metadata
        await self._save()
        
    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for similar chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)[0].reshape(1, -1).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Get chunks and create results
        results = []
        chunk_list = list(self.chunks.values())
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(chunk_list):
                chunk = chunk_list[idx]
                score = 1 / (1 + distance)  # Convert distance to similarity score
                
                # Get parent document
                document = self.documents.get(chunk.document_id)
                
                result = SearchResult(
                    chunk=chunk,
                    score=float(score),
                    document=document
                )
                results.append(result)
                
        return results
        
    async def delete_document(self, doc_id: str):
        """Delete a document and its chunks"""
        if doc_id in self.documents:
            # Remove chunks
            doc = self.documents[doc_id]
            for chunk in doc.chunks:
                if chunk.id in self.chunks:
                    del self.chunks[chunk.id]
                    
            # Remove document
            del self.documents[doc_id]
            
            # Rebuild index
            await self._rebuild_index()
            
    async def _rebuild_index(self):
        """Rebuild the FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_model.embedding_dim)
        
        all_embeddings = []
        for chunk in self.chunks.values():
            if chunk.embedding is not None:
                all_embeddings.append(chunk.embedding)
                
        if all_embeddings:
            embeddings_array = np.array(all_embeddings).astype('float32')
            self.index.add(embeddings_array)
            
        await self._save()
        
    async def _save(self):
        """Save index and metadata"""
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save metadata
        metadata = {
            'documents': self.documents,
            'chunks': self.chunks
        }
        
        async with aiofiles.open(f"{self.index_path}.meta", 'wb') as f:
            await f.write(pickle.dumps(metadata))
            
    def _load_metadata(self):
        """Load metadata"""
        with open(f"{self.index_path}.meta", 'rb') as f:
            metadata = pickle.load(f)
            self.documents = metadata['documents']
            self.chunks = metadata['chunks']


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store"""
    
    def __init__(self, embedding_model: EmbeddingModel, collection_name: str = "knowledge_base"):
        self.embedding_model = embedding_model
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection = self.client.get_or_create_collection(collection_name)
        
    async def add_documents(self, documents: List[Document]):
        """Add documents to ChromaDB"""
        for doc in documents:
            for chunk in doc.chunks:
                # Generate embedding if needed
                if chunk.embedding is None:
                    chunk.embedding = self.embedding_model.embed_text(chunk.content)[0]
                    
                # Add to collection
                self.collection.add(
                    embeddings=[chunk.embedding.tolist()],
                    documents=[chunk.content],
                    metadatas=[{
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.id,
                        **chunk.metadata
                    }],
                    ids=[chunk.id]
                )
                
    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search in ChromaDB"""
        query_embedding = self.embedding_model.embed_text(query)[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            chunk = DocumentChunk(
                id=results['ids'][0][i],
                document_id=results['metadatas'][0][i]['document_id'],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            
            score = 1 / (1 + results['distances'][0][i])
            
            search_results.append(SearchResult(
                chunk=chunk,
                score=score
            ))
            
        return search_results
        
    async def delete_document(self, doc_id: str):
        """Delete document from ChromaDB"""
        # Get all chunks for document
        results = self.collection.get(
            where={"document_id": doc_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])


class KnowledgeBase:
    """Main knowledge base system with multiple stores and retrieval strategies"""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = FAISSVectorStore(self.embedding_model)
        self.chunker = TextChunker()
        self.documents: Dict[str, Document] = {}
        
        # Additional indexes
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = None
        self.doc_index_mapping = {}
        
    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> Document:
        """Add a document to the knowledge base"""
        # Create document
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        # Chunk the document
        chunks = self.chunker.chunk_text(content, metadata)
        
        # Create document object
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            chunks=chunks
        )
        
        # Store document
        self.documents[doc_id] = document
        
        # Add to vector store
        await self.vector_store.add_documents([document])
        
        # Update TF-IDF index
        await self._update_tfidf_index()
        
        return document
        
    async def add_documents_batch(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[Document]:
        """Add multiple documents in batch"""
        added_docs = []
        
        for content, metadata in documents:
            doc = await self.add_document(content, metadata)
            added_docs.append(doc)
            
        return added_docs
        
    async def search(self, query: str, k: int = 5, strategy: str = "hybrid") -> List[SearchResult]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            k: Number of results
            strategy: Search strategy (semantic, keyword, hybrid)
        """
        if strategy == "semantic":
            return await self._semantic_search(query, k)
        elif strategy == "keyword":
            return await self._keyword_search(query, k)
        elif strategy == "hybrid":
            return await self._hybrid_search(query, k)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
            
    async def _semantic_search(self, query: str, k: int) -> List[SearchResult]:
        """Semantic search using embeddings"""
        return await self.vector_store.search(query, k)
        
    async def _keyword_search(self, query: str, k: int) -> List[SearchResult]:
        """Keyword search using TF-IDF"""
        if self.tfidf_matrix is None:
            return []
            
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx in self.doc_index_mapping:
                chunk_id = self.doc_index_mapping[idx]
                chunk = self.vector_store.chunks.get(chunk_id)
                
                if chunk:
                    doc = self.documents.get(chunk.document_id)
                    results.append(SearchResult(
                        chunk=chunk,
                        score=float(similarities[idx]),
                        document=doc
                    ))
                    
        return results
        
    async def _hybrid_search(self, query: str, k: int) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search"""
        # Get results from both methods
        semantic_results = await self._semantic_search(query, k)
        keyword_results = await self._keyword_search(query, k)
        
        # Combine and re-rank
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.chunk.id] = result
            
        # Merge keyword results
        for result in keyword_results:
            if result.chunk.id in combined_results:
                # Average the scores
                combined_results[result.chunk.id].score = (
                    combined_results[result.chunk.id].score + result.score
                ) / 2
            else:
                combined_results[result.chunk.id] = result
                
        # Sort by score and return top k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_results[:k]
        
    async def _update_tfidf_index(self):
        """Update TF-IDF index"""
        # Get all chunk texts
        texts = []
        chunk_ids = []
        
        for chunk_id, chunk in self.vector_store.chunks.items():
            texts.append(chunk.content)
            chunk_ids.append(chunk_id)
            
        if texts:
            # Fit or transform TF-IDF
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Update mapping
            self.doc_index_mapping = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
    async def get_relevant_context(self, query: str, max_tokens: int = 2048) -> str:
        """Get relevant context for a query within token limit"""
        # Search for relevant chunks
        results = await self.search(query, k=10)
        
        # Build context within token limit
        context_parts = []
        current_tokens = 0
        tokenizer = self.chunker.tokenizer
        
        for result in results:
            chunk_tokens = len(tokenizer.encode(result.chunk.content))
            
            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(result.chunk.content)
                current_tokens += chunk_tokens
            else:
                # Truncate if needed
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if meaningful
                    truncated = tokenizer.decode(
                        tokenizer.encode(result.chunk.content)[:remaining_tokens]
                    )
                    context_parts.append(truncated)
                break
                
        return "\n\n".join(context_parts)
        
    async def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Update an existing document"""
        # Delete old version
        await self.vector_store.delete_document(doc_id)
        
        # Add new version
        await self.add_document(content, metadata)
        
    async def delete_document(self, doc_id: str):
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            await self.vector_store.delete_document(doc_id)
            await self._update_tfidf_index()


class ConversationSummarizer:
    """Summarizes conversations for long-term memory"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        from transformers import pipeline
        self.summarizer = pipeline("summarization", model=model_name)
        self.entity_extractor = pipeline("ner", model="dslim/bert-base-NER")
        
    def summarize_conversation(self, messages: List[Dict[str, str]], max_length: int = 150) -> ConversationSummary:
        """Summarize a conversation"""
        # Combine messages
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        # Generate summary
        summary_result = self.summarizer(
            conversation_text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        summary_text = summary_result[0]['summary_text']
        
        # Extract key points
        key_points = self._extract_key_points(messages)
        
        # Extract entities
        entities = self._extract_entities(conversation_text)
        
        # Extract topics
        topics = self._extract_topics(conversation_text)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(messages)
        
        # Create summary object
        conv_id = hashlib.md5(conversation_text.encode()).hexdigest()
        summary = ConversationSummary(
            id=f"summary_{conv_id}",
            conversation_id=conv_id,
            summary=summary_text,
            key_points=key_points,
            entities=entities,
            topics=topics,
            sentiment=sentiment
        )
        
        return summary
        
    def _extract_key_points(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extract key points from conversation"""
        key_points = []
        
        # Look for questions and their answers
        for i, msg in enumerate(messages):
            if msg['role'] == 'user' and '?' in msg['content']:
                # This is a question
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    # Found Q&A pair
                    key_points.append(f"Q: {msg['content'][:100]}... A: {messages[i + 1]['content'][:100]}...")
                    
        return key_points[:5]  # Limit to 5 key points
        
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities"""
        entities = self.entity_extractor(text)
        
        # Group by entity type
        grouped_entities = defaultdict(list)
        for entity in entities:
            grouped_entities[entity['entity']].append(entity['word'])
            
        # Format results
        formatted_entities = []
        for entity_type, words in grouped_entities.items():
            formatted_entities.append({
                'type': entity_type,
                'values': list(set(words))  # Unique values
            })
            
        return formatted_entities
        
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics using simple keyword extraction"""
        # Simple topic extraction based on noun phrases
        # In production, use more sophisticated topic modeling
        from collections import Counter
        import nltk
        
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.corpus import stopwords
            
            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract nouns
            stop_words = set(stopwords.words('english'))
            nouns = [
                word for word, pos in pos_tags
                if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and word not in stop_words and len(word) > 3
            ]
            
            # Count frequency
            noun_freq = Counter(nouns)
            
            # Return top topics
            return [noun for noun, _ in noun_freq.most_common(5)]
            
        except:
            # Fallback to simple word frequency
            words = text.lower().split()
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(5) if len(word) > 4]
            
    def _analyze_sentiment(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """Analyze conversation sentiment"""
        # Simple sentiment based on message content
        positive_words = {'good', 'great', 'excellent', 'happy', 'thanks', 'perfect'}
        negative_words = {'bad', 'poor', 'terrible', 'unhappy', 'problem', 'issue'}
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for msg in messages:
            words = set(msg['content'].lower().split())
            positive_count += len(words & positive_words)
            negative_count += len(words & negative_words)
            total_words += len(words)
            
        if total_words > 0:
            return {
                'positive': positive_count / total_words,
                'negative': negative_count / total_words,
                'neutral': 1 - (positive_count + negative_count) / total_words
            }
        else:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}


class LearningEngine:
    """Implements learning and adaptation capabilities"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.user_preferences: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.feedback_scores: Dict[str, float] = {}
        self.learned_patterns: Dict[str, List[str]] = defaultdict(list)
        
    async def learn_from_interaction(self, user_input: str, response: str, feedback: Optional[float] = None):
        """Learn from a single interaction"""
        interaction = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'response': response,
            'feedback': feedback
        }
        
        self.interaction_history.append(interaction)
        
        # Learn patterns
        await self._learn_patterns(user_input, response)
        
        # Update preferences
        await self._update_preferences(user_input, response, feedback)
        
        # Store good interactions in knowledge base
        if feedback and feedback > 0.8:
            await self._store_successful_interaction(user_input, response)
            
    async def _learn_patterns(self, user_input: str, response: str):
        """Learn common patterns from interactions"""
        # Extract intent/pattern
        input_lower = user_input.lower()
        
        # Simple pattern matching
        patterns = {
            'greeting': ['hi', 'hello', 'hey', 'good morning'],
            'question': ['what', 'how', 'why', 'when', 'where'],
            'request': ['can you', 'could you', 'please', 'would you'],
            'feedback': ['thanks', 'thank you', 'great', 'good job']
        }
        
        for pattern_type, keywords in patterns.items():
            if any(keyword in input_lower for keyword in keywords):
                self.learned_patterns[pattern_type].append(response)
                
    async def _update_preferences(self, user_input: str, response: str, feedback: Optional[float]):
        """Update user preferences based on feedback"""
        if feedback is not None:
            # Extract features from the interaction
            features = {
                'response_length': len(response.split()),
                'formal_tone': 'please' in response.lower() or 'would' in response.lower(),
                'detailed': len(response) > 200,
                'technical': any(term in response.lower() for term in ['algorithm', 'function', 'system'])
            }
            
            # Update preferences based on feedback
            for feature, value in features.items():
                if feature not in self.user_preferences:
                    self.user_preferences[feature] = {'score': 0.5, 'count': 0}
                    
                # Update score with exponential moving average
                alpha = 0.3
                old_score = self.user_preferences[feature]['score']
                self.user_preferences[feature]['score'] = alpha * feedback + (1 - alpha) * old_score
                self.user_preferences[feature]['count'] += 1
                
    async def _store_successful_interaction(self, user_input: str, response: str):
        """Store successful interactions in knowledge base"""
        # Create a document from the interaction
        content = f"Q: {user_input}\nA: {response}"
        metadata = {
            'type': 'learned_interaction',
            'timestamp': datetime.now().isoformat(),
            'success_score': 0.9
        }
        
        await self.knowledge_base.add_document(content, metadata)
        
    def get_adapted_parameters(self) -> Dict[str, Any]:
        """Get adapted parameters based on learned preferences"""
        params = {}
        
        # Adapt response length
        if 'response_length' in self.user_preferences:
            pref = self.user_preferences['response_length']
            if pref['score'] > 0.7:
                params['max_length'] = 300  # Longer responses
            elif pref['score'] < 0.3:
                params['max_length'] = 100  # Shorter responses
                
        # Adapt formality
        if 'formal_tone' in self.user_preferences:
            pref = self.user_preferences['formal_tone']
            params['formal'] = pref['score'] > 0.5
            
        # Adapt detail level
        if 'detailed' in self.user_preferences:
            pref = self.user_preferences['detailed']
            params['detail_level'] = 'high' if pref['score'] > 0.6 else 'low'
            
        return params
        
    def get_personalized_prompt_additions(self) -> str:
        """Get personalized additions to system prompt based on learning"""
        additions = []
        
        # Add tone preferences
        if self.user_preferences.get('formal_tone', {}).get('score', 0.5) > 0.7:
            additions.append("Use a formal and professional tone.")
        elif self.user_preferences.get('formal_tone', {}).get('score', 0.5) < 0.3:
            additions.append("Use a casual and friendly tone.")
            
        # Add detail preferences
        if self.user_preferences.get('detailed', {}).get('score', 0.5) > 0.7:
            additions.append("Provide detailed and comprehensive answers.")
        elif self.user_preferences.get('detailed', {}).get('score', 0.5) < 0.3:
            additions.append("Keep answers concise and to the point.")
            
        return " ".join(additions)


class RAGEngine:
    """Main RAG engine integrating all components"""
    
    def __init__(self, base_model_name: str = "gpt2"):
        self.embedding_model = EmbeddingModel()
        self.knowledge_base = KnowledgeBase(self.embedding_model)
        self.summarizer = ConversationSummarizer()
        self.learning_engine = LearningEngine(self.knowledge_base)
        self.base_model_name = base_model_name
        
    async def generate_with_retrieval(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate response using retrieval-augmented generation"""
        # Get relevant context
        context = await self.knowledge_base.get_relevant_context(query, max_tokens=1024)
        
        # Get adapted parameters
        adapted_params = self.learning_engine.get_adapted_parameters()
        
        # Get personalized prompt additions
        prompt_additions = self.learning_engine.get_personalized_prompt_additions()
        
        # Build augmented prompt
        augmented_prompt = self._build_augmented_prompt(
            query,
            context,
            conversation_history,
            prompt_additions
        )
        
        # Generate response (placeholder - integrate with actual model)
        response = f"Based on the context: {context[:200]}... I can help you with: {query}"
        
        # Store interaction for learning
        await self.learning_engine.learn_from_interaction(query, response)
        
        return {
            'response': response,
            'context_used': context,
            'adapted_parameters': adapted_params,
            'sources': await self._get_source_documents(query)
        }
        
    def _build_augmented_prompt(self, query: str, context: str, history: List[Dict[str, str]], additions: str) -> str:
        """Build prompt with retrieved context"""
        prompt_parts = []
        
        # System prompt with additions
        system_prompt = f"You are a helpful AI assistant with access to a knowledge base. {additions}"
        prompt_parts.append(f"System: {system_prompt}")
        
        # Add context
        if context:
            prompt_parts.append(f"\nRelevant Context:\n{context}")
            
        # Add conversation history
        if history:
            prompt_parts.append("\nConversation History:")
            for msg in history[-5:]:  # Last 5 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
                
        # Add current query
        prompt_parts.append(f"\nUser: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
        
    async def _get_source_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get source documents for attribution"""
        results = await self.knowledge_base.search(query, k=k)
        
        sources = []
        for result in results:
            if result.document:
                sources.append({
                    'id': result.document.id,
                    'content_preview': result.chunk.content[:200] + "...",
                    'score': result.score,
                    'metadata': result.document.metadata
                })
                
        return sources
        
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to the base"""
        return await self.knowledge_base.add_document(content, metadata)
        
    async def summarize_conversation(self, messages: List[Dict[str, str]]) -> ConversationSummary:
        """Summarize a conversation"""
        summary = self.summarizer.summarize_conversation(messages)
        
        # Store summary in knowledge base
        await self.knowledge_base.add_document(
            summary.summary,
            {
                'type': 'conversation_summary',
                'conversation_id': summary.conversation_id,
                'key_points': summary.key_points,
                'entities': summary.entities,
                'topics': summary.topics
            }
        )
        
        return summary
        
    async def provide_feedback(self, query: str, response: str, score: float):
        """Provide feedback for learning"""
        await self.learning_engine.learn_from_interaction(query, response, score)
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning engine"""
        return {
            'user_preferences': self.learning_engine.user_preferences,
            'total_interactions': len(self.learning_engine.interaction_history),
            'learned_patterns': dict(self.learning_engine.learned_patterns),
            'average_feedback': np.mean([
                i['feedback'] for i in self.learning_engine.interaction_history
                if i['feedback'] is not None
            ]) if any(i['feedback'] is not None for i in self.learning_engine.interaction_history) else None
        }


# Example usage
if __name__ == "__main__":
    async def test_rag():
        # Initialize RAG engine
        rag = RAGEngine()
        
        # Add some knowledge
        await rag.add_knowledge(
            "Python is a high-level programming language known for its simplicity and readability.",
            {"source": "tutorial", "topic": "programming"}
        )
        
        await rag.add_knowledge(
            "Machine learning is a subset of AI that enables systems to learn from data.",
            {"source": "textbook", "topic": "AI"}
        )
        
        # Test retrieval
        result = await rag.generate_with_retrieval("Tell me about Python programming")
        print(f"Response: {result['response']}")
        print(f"Context used: {result['context_used'][:100]}...")
        
        # Test conversation summarization
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Sure! Image recognition is a common example..."}
        ]
        
        summary = await rag.summarize_conversation(messages)
        print(f"\nSummary: {summary.summary}")
        print(f"Key points: {summary.key_points}")
        print(f"Topics: {summary.topics}")
        
    # Run test
    asyncio.run(test_rag())
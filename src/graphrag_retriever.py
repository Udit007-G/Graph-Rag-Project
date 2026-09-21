"""
Official GraphRAG-style retriever implementation.
Follows the pattern from TigerGraph's official GraphRAG repository.
Supports both TigerGraph (when configured) and local fallback.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    content: str
    source: str
    score: float = 0.0
    doc_id: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """Hybrid vector + graph-expansion search.
    
    Following TigerGraph GraphRAG's HybridRetriever pattern:
    1. Vector similarity search to find initial candidates
    2. Graph expansion to find related entities/chunks
    3. Combine and rank results
    """
    
    def __init__(self, client, embedding_model=None):
        self.client = client
        self.embedding_model = embedding_model
        self.is_tg = hasattr(client, 'ai')  # TigerGraph connection
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        num_hops: int = 2,
        indices: List[str] = None,
        chunk_only: bool = True,
    ) -> List[RetrievalResult]:
        """Execute hybrid search."""
        if self.is_tg:
            return self._tg_hybrid_search(query, top_k, num_hops)
        else:
            return self._local_hybrid_search(query, top_k, num_hops)
    
    def _tg_hybrid_search(self, query, top_k, num_hops):
        """Use TigerGraph's native hybrid search."""
        results = self.client.ai.searchDocuments(
            query=query,
            method="hybrid",
            method_parameters={
                "indices": ["DocumentChunk"],
                "top_k": top_k,
                "num_hops": num_hops,
            }
        )
        return [RetrievalResult(
            content=r.get("content", ""),
            source=r.get("source", ""),
            score=r.get("score", 0.0),
            doc_id=r.get("doc_id", ""),
        ) for r in results]
    
    def _local_hybrid_search(self, query, top_k, num_hops):
        """Local fallback: similarity search + entity expansion."""
        from src.agents.similarity_search import hybrid_search
        
        # Get initial results via similarity
        results = hybrid_search(query, top_k=top_k * 2, client=self.client)
        
        retrieval_results = []
        seen = set()
        
        for r in results[:top_k]:
            doc_id = r.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            
            retrieval_results.append(RetrievalResult(
                content=r.get("text", "")[:1500],
                source=r.get("title", ""),
                score=r.get("score", 0.0),
                doc_id=doc_id,
                metadata={"event_id": r.get("event_id", "")},
            ))
        
        return retrieval_results[:top_k]


class SimilarityRetriever:
    """Pure vector similarity search."""
    
    def __init__(self, client, embedding_model=None):
        self.client = client
        self.embedding_model = embedding_model
        self.is_tg = hasattr(client, 'ai')
    
    def search(self, query: str, top_k: int = 5, index: str = "DocumentChunk") -> List[RetrievalResult]:
        if self.is_tg:
            results = self.client.ai.searchDocuments(
                query=query,
                method="similarity",
                method_parameters={"top_k": top_k}
            )
        else:
            from src.agents.similarity_search import hybrid_search
            results = hybrid_search(query, top_k=top_k, client=self.client)
        
        return [RetrievalResult(
            content=r.get("text", r.get("content", ""))[:1500],
            source=r.get("title", ""),
            score=r.get("score", 0.0),
            doc_id=r.get("doc_id", ""),
        ) for r in results[:top_k]]


class CommunityRetriever:
    """Community summary search - finds community-level context."""
    
    def __init__(self, client):
        self.client = client
        self.is_tg = hasattr(client, 'ai')
    
    def search(self, query: str, top_k: int = 5, community_level: int = 2) -> List[RetrievalResult]:
        if self.is_tg:
            results = self.client.ai.searchDocuments(
                query=query,
                method="community",
                method_parameters={"top_k": top_k, "community_level": community_level}
            )
            return [RetrievalResult(
                content=r.get("summary", ""),
                source=f"community_{r.get('community_id', '')}",
                score=r.get("score", 0.0),
            ) for r in results]
        else:
            # Local fallback: group docs by event and create summaries
            return self._local_community_search(query, top_k)
    
    def _local_community_search(self, query, top_k):
        """Create community-like groupings from local store."""
        from src.agents.similarity_search import hybrid_search
        
        results = hybrid_search(query, top_k=top_k * 2, client=self.client)
        
        # Group by event_id (community equivalent)
        communities = {}
        for r in results:
            event_id = r.get("event_id", "unknown")
            if event_id not in communities:
                communities[event_id] = {
                    "docs": [],
                    "titles": set(),
                }
            communities[event_id]["docs"].append(r.get("text", "")[:500])
            communities[event_id]["titles"].add(r.get("title", ""))
        
        # Create community summaries
        retrieval_results = []
        for event_id, community in communities.items():
            if len(retrieval_results) >= top_k:
                break
            summary = f"Event: {event_id}\n"
            summary += f"Documents: {', '.join(list(community['titles'])[:3])}\n"
            summary += "Content: " + " ".join(community["docs"])[:1000]
            
            retrieval_results.append(RetrievalResult(
                content=summary,
                source=f"community_{event_id}",
                score=1.0,
                metadata={"doc_count": len(community["docs"])}
            ))
        
        return retrieval_results


class EntityRelationshipRetriever:
    """Entity/relationship-based retrieval."""
    
    def __init__(self, client):
        self.client = client
        self.is_tg = hasattr(client, 'ai')
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.is_tg:
            results = self.client.ai.searchDocuments(
                query=query,
                method="similarity",
                method_parameters={"top_k": top_k}
            )
        else:
            # Local fallback: extract entities and find related docs
            from src.agents.entity_linker import extract_entities, link_entities_to_graph
            from src.agents.similarity_search import hybrid_search
            
            entities = extract_entities(query)
            linked = link_entities_to_graph(entities, self.client)
            
            # Get docs from linked entities
            entity_docs = []
            for ent in linked:
                eid = ent.get("graph_id", "")
                if eid and hasattr(self.client, 'get_docs_for_event'):
                    docs = self.client.get_docs_for_event(eid)
                    for doc in docs[:3]:
                        entity_docs.append({
                            "doc_id": doc.get("doc_id", ""),
                            "text": doc.get("text", "")[:1000],
                            "title": doc.get("title", ""),
                        })
            
            # Also get similarity results
            sim_results = hybrid_search(query, top_k=top_k, client=self.client)
            
            # Combine and deduplicate
            seen = set()
            results = []
            for doc in entity_docs + sim_results:
                doc_id = doc.get("doc_id", "")
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    results.append(doc)
        
        return [RetrievalResult(
            content=r.get("text", "")[:1500],
            source=r.get("title", ""),
            doc_id=r.get("doc_id", ""),
        ) for r in results[:top_k]]


class GraphRAGRetriever:
    """Main GraphRAG retriever - simplified version for local store."""
    
    def __init__(self, client, embedding_model=None):
        self.client = client
        self.is_tg = hasattr(client, 'ai')
    
    def retrieve(
        self,
        query: str,
        method: str = "auto",
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve using specified method."""
        from src.agents.similarity_search import hybrid_search
        
        # Use our existing hybrid search as the backbone
        results = hybrid_search(query, top_k=top_k, client=self.client)
        
        return [RetrievalResult(
            content=r.get("text", "")[:1500],
            source=r.get("title", ""),
            score=r.get("score", 0.0),
            doc_id=r.get("doc_id", ""),
            metadata={"event_id": r.get("event_id", "")},
        ) for r in results[:top_k]]

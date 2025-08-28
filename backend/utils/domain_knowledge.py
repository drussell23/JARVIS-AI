import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
from dataclasses import dataclass, field
from collections import defaultdict
import pickle

@dataclass
class DomainKnowledge:
    """Container for domain-specific knowledge"""
    domain: str
    facts: List[str] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    terminology: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

class DomainKnowledgeBank:
    """Manages domain-specific knowledge for multiple domains"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.domains: Dict[str, DomainKnowledge] = {}
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Index for fast retrieval
        self.knowledge_index = None
        self.index_to_knowledge = {}
        
    def add_domain(self, domain_name: str, knowledge: DomainKnowledge):
        """Add domain knowledge"""
        # Generate embeddings for facts
        if knowledge.facts:
            embeddings = self.embedding_model.encode(knowledge.facts)
            knowledge.embeddings = embeddings
            
        self.domains[domain_name] = knowledge
        self._rebuild_index()
        
    def load_from_file(self, file_path: str):
        """Load domain knowledge from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for domain_name, domain_data in data.items():
            knowledge = DomainKnowledge(
                domain=domain_name,
                facts=domain_data.get('facts', []),
                rules=domain_data.get('rules', []),
                examples=domain_data.get('examples', []),
                terminology=domain_data.get('terminology', {}),
                constraints=domain_data.get('constraints', [])
            )
            self.add_domain(domain_name, knowledge)
            
    def save_to_file(self, file_path: str):
        """Save domain knowledge to file"""
        data = {}
        for domain_name, knowledge in self.domains.items():
            data[domain_name] = {
                'facts': knowledge.facts,
                'rules': knowledge.rules,
                'examples': knowledge.examples,
                'terminology': knowledge.terminology,
                'constraints': knowledge.constraints
            }
            
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _rebuild_index(self):
        """Rebuild FAISS index for all knowledge"""
        all_embeddings = []
        self.index_to_knowledge = {}
        idx = 0
        
        for domain_name, knowledge in self.domains.items():
            if knowledge.embeddings is not None:
                for i, embedding in enumerate(knowledge.embeddings):
                    all_embeddings.append(embedding)
                    self.index_to_knowledge[idx] = (domain_name, i)
                    idx += 1
                    
        if all_embeddings:
            embeddings_array = np.array(all_embeddings).astype('float32')
            self.knowledge_index = faiss.IndexFlatL2(self.embedding_dim)
            self.knowledge_index.add(embeddings_array)
            
    def retrieve_relevant_knowledge(
        self, 
        query: str, 
        domain: Optional[str] = None,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant knowledge for a query"""
        if self.knowledge_index is None:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in index
        distances, indices = self.knowledge_index.search(
            query_embedding.astype('float32'), 
            k * 2 if domain else k
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx in self.index_to_knowledge:
                domain_name, fact_idx = self.index_to_knowledge[idx]
                
                # Filter by domain if specified
                if domain and domain_name != domain:
                    continue
                    
                fact = self.domains[domain_name].facts[fact_idx]
                score = 1 / (1 + distance)  # Convert distance to similarity
                results.append((fact, score))
                
        return results[:k]
        
    def get_domain_rules(self, domain: str) -> List[Dict[str, Any]]:
        """Get rules for a specific domain"""
        if domain in self.domains:
            return self.domains[domain].rules
        return []
        
    def get_domain_examples(self, domain: str, intent: Optional[str] = None) -> List[Dict[str, str]]:
        """Get examples for a domain, optionally filtered by intent"""
        if domain not in self.domains:
            return []
            
        examples = self.domains[domain].examples
        
        if intent:
            # Filter examples by intent
            examples = [ex for ex in examples if ex.get('intent') == intent]
            
        return examples
        
    def get_terminology(self, domain: str, term: str) -> Optional[str]:
        """Get definition of a domain-specific term"""
        if domain in self.domains:
            return self.domains[domain].terminology.get(term)
        return None
        
    def check_constraints(self, domain: str, action: str) -> List[str]:
        """Check constraints for a domain action"""
        violations = []
        
        if domain in self.domains:
            for constraint in self.domains[domain].constraints:
                # Simple constraint checking - in practice, use more sophisticated logic
                if self._violates_constraint(action, constraint):
                    violations.append(constraint)
                    
        return violations
        
    def _violates_constraint(self, action: str, constraint: str) -> bool:
        """Check if an action violates a constraint"""
        # Placeholder - implement actual constraint checking logic
        return False

class DomainAdapter(nn.Module):
    """Neural adapter for domain-specific knowledge"""
    
    def __init__(
        self,
        base_model_dim: int,
        num_domains: int,
        adapter_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_domains = num_domains
        self.adapter_size = adapter_size
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, adapter_size)
        
        # Adapter layers
        self.down_project = nn.Linear(base_model_dim, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, base_model_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Domain-specific parameters
        self.domain_specific = nn.ModuleList([
            nn.Linear(adapter_size, adapter_size) for _ in range(num_domains)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(base_model_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        domain_id: torch.Tensor
    ) -> torch.Tensor:
        """Apply domain-specific adaptation"""
        residual = hidden_states
        
        # Down project
        hidden = self.down_project(hidden_states)
        hidden = self.activation(hidden)
        
        # Apply domain-specific transformation
        batch_size = hidden.size(0)
        domain_output = torch.zeros_like(hidden)
        
        for i in range(batch_size):
            domain_idx = domain_id[i].item()
            if 0 <= domain_idx < self.num_domains:
                domain_hidden = self.domain_specific[domain_idx](hidden[i])
                domain_output[i] = domain_hidden
            else:
                domain_output[i] = hidden[i]
                
        # Add domain embedding
        domain_emb = self.domain_embeddings(domain_id)
        domain_emb = domain_emb.unsqueeze(1).expand(-1, hidden.size(1), -1)
        hidden = domain_output + domain_emb
        
        # Up project
        hidden = self.up_project(hidden)
        hidden = self.dropout(hidden)
        
        # Residual connection
        output = self.layer_norm(hidden + residual)
        
        return output

class KnowledgeIntegrator:
    """Integrates domain knowledge into model predictions"""
    
    def __init__(
        self,
        knowledge_bank: DomainKnowledgeBank,
        model_dim: int = 768
    ):
        self.knowledge_bank = knowledge_bank
        self.model_dim = model_dim
        
        # Knowledge encoder
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(knowledge_bank.embedding_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
    def integrate_knowledge(
        self,
        query: str,
        hidden_states: torch.Tensor,
        domain: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate relevant knowledge into hidden states"""
        # Retrieve relevant knowledge
        relevant_facts = self.knowledge_bank.retrieve_relevant_knowledge(
            query, domain, k=3
        )
        
        if not relevant_facts:
            return hidden_states, {"knowledge_used": False}
            
        # Encode knowledge
        knowledge_texts = [fact for fact, _ in relevant_facts]
        knowledge_embeddings = self.knowledge_bank.embedding_model.encode(knowledge_texts)
        knowledge_embeddings = torch.tensor(knowledge_embeddings).float()
        
        # Transform knowledge to model space
        knowledge_vectors = self.knowledge_encoder(knowledge_embeddings)
        
        # Aggregate knowledge
        knowledge_vector = knowledge_vectors.mean(dim=0)
        
        # Integrate with hidden states
        # Simple addition - can be replaced with more sophisticated fusion
        enhanced_states = hidden_states + knowledge_vector.unsqueeze(0).unsqueeze(0)
        
        metadata = {
            "knowledge_used": True,
            "facts_retrieved": knowledge_texts,
            "domain": domain
        }
        
        return enhanced_states, metadata

class DomainSpecificHeads(nn.Module):
    """Domain-specific output heads for specialized tasks"""
    
    def __init__(
        self,
        base_model_dim: int,
        vocab_size: int,
        num_domains: int
    ):
        super().__init__()
        self.num_domains = num_domains
        
        # Shared base projection
        self.base_projection = nn.Linear(base_model_dim, base_model_dim)
        
        # Domain-specific heads
        self.domain_heads = nn.ModuleList([
            nn.Linear(base_model_dim, vocab_size) for _ in range(num_domains)
        ])
        
        # Default head for unknown domains
        self.default_head = nn.Linear(base_model_dim, vocab_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply domain-specific output head"""
        # Base projection
        hidden = self.base_projection(hidden_states)
        
        # Apply domain-specific heads
        batch_size, seq_len = hidden.size()[:2]
        vocab_size = self.default_head.out_features
        
        output = torch.zeros(batch_size, seq_len, vocab_size, device=hidden.device)
        
        for i in range(batch_size):
            domain_idx = domain_ids[i].item()
            
            if 0 <= domain_idx < self.num_domains:
                output[i] = self.domain_heads[domain_idx](hidden[i])
            else:
                output[i] = self.default_head(hidden[i])
                
        return output

# Example domain knowledge data
EXAMPLE_DOMAINS = {
    "medical": {
        "facts": [
            "Aspirin is commonly used for pain relief and reducing inflammation",
            "Blood pressure should be monitored regularly in hypertensive patients",
            "Diabetes requires careful management of blood sugar levels",
            "Antibiotics are not effective against viral infections"
        ],
        "rules": [
            {
                "condition": "patient has allergy to penicillin",
                "action": "avoid prescribing penicillin-based antibiotics",
                "severity": "critical"
            }
        ],
        "examples": [
            {
                "input": "Patient complains of headache",
                "response": "I understand you're experiencing a headache. Can you describe the pain?",
                "intent": "symptom_inquiry"
            }
        ],
        "terminology": {
            "hypertension": "High blood pressure",
            "tachycardia": "Rapid heart rate",
            "analgesic": "Pain reliever"
        },
        "constraints": [
            "Never provide specific medical diagnosis without proper examination",
            "Always recommend consulting healthcare provider for serious symptoms"
        ]
    },
    "technical_support": {
        "facts": [
            "Restarting the device often resolves temporary software issues",
            "Regular software updates improve security and performance",
            "Backing up data prevents loss in case of system failure",
            "Network connectivity issues can often be resolved by resetting the router"
        ],
        "rules": [
            {
                "condition": "user reports data loss",
                "action": "first check for backups before attempting recovery",
                "priority": "high"
            }
        ],
        "examples": [
            {
                "input": "My computer is running slowly",
                "response": "I can help you improve your computer's performance. Let's start by checking what might be causing the slowdown.",
                "intent": "troubleshooting"
            }
        ],
        "terminology": {
            "RAM": "Random Access Memory - temporary storage for running programs",
            "CPU": "Central Processing Unit - the brain of the computer",
            "bandwidth": "The amount of data that can be transmitted over a network"
        },
        "constraints": [
            "Always ensure data backup before major system changes",
            "Verify user permissions before accessing system settings"
        ]
    }
}

# Example usage
if __name__ == "__main__":
    # Create knowledge bank
    kb = DomainKnowledgeBank()
    
    # Add example domains
    for domain_name, domain_data in EXAMPLE_DOMAINS.items():
        knowledge = DomainKnowledge(
            domain=domain_name,
            facts=domain_data["facts"],
            rules=domain_data["rules"],
            examples=domain_data["examples"],
            terminology=domain_data["terminology"],
            constraints=domain_data["constraints"]
        )
        kb.add_domain(domain_name, knowledge)
        
    # Test retrieval
    results = kb.retrieve_relevant_knowledge("headache pain", domain="medical")
    print("Retrieved knowledge:")
    for fact, score in results:
        print(f"- {fact} (score: {score:.3f})")
        
    # Create domain adapter
    adapter = DomainAdapter(
        base_model_dim=768,
        num_domains=10,
        adapter_size=256
    )
    
    # Test adapter
    hidden_states = torch.randn(2, 10, 768)
    domain_ids = torch.tensor([0, 1])
    output = adapter(hidden_states, domain_ids)
    print(f"\nAdapter output shape: {output.shape}")
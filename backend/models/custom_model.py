import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union, List, Dict, Any
import math
import numpy as np
from dataclasses import dataclass
import json

class CustomChatbotConfig(PretrainedConfig):
    """Configuration for custom chatbot model"""
    model_type = "custom_chatbot"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Custom parameters
        use_domain_embeddings: bool = True,
        domain_embedding_size: int = 128,
        num_domains: int = 10,
        use_intent_aware_attention: bool = True,
        intent_embedding_size: int = 64,
        num_intents: int = 20,
        use_context_fusion: bool = True,
        context_window_size: int = 5,
        use_knowledge_integration: bool = True,
        knowledge_dim: int = 256,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Custom parameters
        self.use_domain_embeddings = use_domain_embeddings
        self.domain_embedding_size = domain_embedding_size
        self.num_domains = num_domains
        self.use_intent_aware_attention = use_intent_aware_attention
        self.intent_embedding_size = intent_embedding_size
        self.num_intents = num_intents
        self.use_context_fusion = use_context_fusion
        self.context_window_size = context_window_size
        self.use_knowledge_integration = use_knowledge_integration
        self.knowledge_dim = knowledge_dim

class DomainAwareEmbedding(nn.Module):
    """Domain-aware embedding layer"""
    
    def __init__(self, config: CustomChatbotConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        if config.use_domain_embeddings:
            self.domain_embeddings = nn.Embedding(config.num_domains, config.domain_embedding_size)
            self.domain_projection = nn.Linear(
                config.hidden_size + config.domain_embedding_size, 
                config.hidden_size
            )
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        
        # Add domain embeddings if available
        if hasattr(self, 'domain_embeddings') and domain_ids is not None:
            domain_emb = self.domain_embeddings(domain_ids)
            # Expand domain embedding to match sequence length
            domain_emb = domain_emb.unsqueeze(1).expand(-1, seq_length, -1)
            embeddings = torch.cat([embeddings, domain_emb], dim=-1)
            embeddings = self.domain_projection(embeddings)
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class IntentAwareAttention(nn.Module):
    """Intent-aware multi-head attention"""
    
    def __init__(self, config: CustomChatbotConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        if config.use_intent_aware_attention:
            self.intent_embeddings = nn.Embedding(config.num_intents, config.intent_embedding_size)
            self.intent_projection = nn.Linear(config.intent_embedding_size, self.all_head_size)
            
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        intent_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Add intent-aware bias if available
        if hasattr(self, 'intent_embeddings') and intent_ids is not None:
            intent_emb = self.intent_embeddings(intent_ids)
            intent_bias = self.intent_projection(intent_emb)
            intent_bias = self.transpose_for_scores(intent_bias.unsqueeze(1))
            # Add intent bias to attention scores
            attention_scores = attention_scores + intent_bias.mean(dim=2, keepdim=True)
            
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class KnowledgeIntegrationLayer(nn.Module):
    """Integrates external knowledge into the model"""
    
    def __init__(self, config: CustomChatbotConfig):
        super().__init__()
        self.knowledge_projection = nn.Linear(config.knowledge_dim, config.hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        knowledge_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if knowledge_vectors is None:
            return hidden_states
            
        # Project knowledge to hidden size
        knowledge_proj = self.knowledge_projection(knowledge_vectors)
        
        # Calculate fusion gate
        gate_input = torch.cat([hidden_states, knowledge_proj], dim=-1)
        gate = self.fusion_gate(gate_input)
        
        # Fuse knowledge with hidden states
        fused = gate * hidden_states + (1 - gate) * knowledge_proj
        fused = self.layer_norm(fused)
        
        return fused

class CustomTransformerBlock(nn.Module):
    """Custom transformer block with domain and intent awareness"""
    
    def __init__(self, config: CustomChatbotConfig):
        super().__init__()
        self.attention = IntentAwareAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Knowledge integration
        if config.use_knowledge_integration:
            self.knowledge_integration = KnowledgeIntegrationLayer(config)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        intent_ids: Optional[torch.Tensor] = None,
        knowledge_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask, intent_ids)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_layer_norm(attention_output + hidden_states)
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        hidden_states = self.output_layer_norm(layer_output + hidden_states)
        
        # Knowledge integration
        if hasattr(self, 'knowledge_integration'):
            hidden_states = self.knowledge_integration(hidden_states, knowledge_vectors)
            
        return hidden_states

class CustomChatbotModel(PreTrainedModel):
    """Custom chatbot model with domain-specific enhancements"""
    
    config_class = CustomChatbotConfig
    
    def __init__(self, config: CustomChatbotConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embeddings = DomainAwareEmbedding(config)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            CustomTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Context fusion for conversation history
        if config.use_context_fusion:
            self.context_fusion = nn.LSTM(
                config.hidden_size,
                config.hidden_size // 2,
                bidirectional=True,
                batch_first=True
            )
            
        # Initialize weights
        self.init_weights()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        intent_ids: Optional[torch.Tensor] = None,
        knowledge_vectors: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids, domain_ids=domain_ids)
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = self._prepare_attention_mask(attention_mask)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                intent_ids=intent_ids,
                knowledge_vectors=knowledge_vectors
            )
            
        # Apply context fusion if enabled
        if hasattr(self, 'context_fusion') and self.config.use_context_fusion:
            hidden_states, _ = self.context_fusion(hidden_states)
            
        # Get logits
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None
        )
        
    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare attention mask for transformer"""
        # Create causal mask
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=attention_mask.device),
            diagonal=1
        ).bool()
        
        # Combine with padding mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Apply causal mask
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.masked_fill(causal_mask, 0)
        
        # Convert to attention scores mask
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        return attention_mask
        
    def generate_with_knowledge(
        self,
        input_ids: torch.Tensor,
        knowledge_context: str,
        domain_id: int = 0,
        intent_id: int = 0,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """Generate response with knowledge integration"""
        # Encode knowledge context
        if hasattr(self, 'knowledge_encoder'):
            knowledge_vectors = self.knowledge_encoder(knowledge_context)
        else:
            # Simple embedding as fallback
            knowledge_vectors = torch.randn(
                1, 
                input_ids.size(1), 
                self.config.knowledge_dim,
                device=input_ids.device
            )
            
        # Prepare domain and intent tensors
        batch_size = input_ids.size(0)
        domain_ids = torch.tensor([domain_id] * batch_size, device=input_ids.device)
        intent_ids = torch.tensor([intent_id] * batch_size, device=input_ids.device)
        
        # Generate with custom parameters
        outputs = self.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            domain_ids=domain_ids,
            intent_ids=intent_ids,
            knowledge_vectors=knowledge_vectors,
            **kwargs
        )
        
        return outputs

class ModelBuilder:
    """Utility class for building custom models"""
    
    @staticmethod
    def create_small_model() -> CustomChatbotModel:
        """Create a small model for testing"""
        config = CustomChatbotConfig(
            vocab_size=30000,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024
        )
        return CustomChatbotModel(config)
        
    @staticmethod
    def create_base_model() -> CustomChatbotModel:
        """Create a base model"""
        config = CustomChatbotConfig(
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        return CustomChatbotModel(config)
        
    @staticmethod
    def create_large_model() -> CustomChatbotModel:
        """Create a large model"""
        config = CustomChatbotConfig(
            vocab_size=50257,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096
        )
        return CustomChatbotModel(config)
        
    @staticmethod
    def from_pretrained_with_custom_layers(
        model_name: str,
        add_domain_awareness: bool = True,
        add_intent_awareness: bool = True,
        add_knowledge_integration: bool = True
    ) -> CustomChatbotModel:
        """Load a pretrained model and add custom layers"""
        # This would load a pretrained model and add custom layers
        # For now, return a new model
        config = CustomChatbotConfig(
            use_domain_embeddings=add_domain_awareness,
            use_intent_aware_attention=add_intent_awareness,
            use_knowledge_integration=add_knowledge_integration
        )
        return CustomChatbotModel(config)

# Example usage
if __name__ == "__main__":
    # Create a small model for testing
    model = ModelBuilder.create_small_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    outputs = model(input_ids)
    print(f"Output shape: {outputs.logits.shape}")
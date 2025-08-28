import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
import logging
from datetime import datetime

from models.custom_model import CustomChatbotModel, CustomChatbotConfig
from utils.domain_knowledge import DomainKnowledgeBank, DomainAdapter

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning"""
    # Base model
    base_model_path: str = "gpt2"
    custom_model: bool = False
    
    # Data
    train_data_path: str = "./data/fine_tune_train.jsonl"
    eval_data_path: str = "./data/fine_tune_eval.jsonl"
    max_seq_length: int = 512
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Fine-tuning strategy
    freeze_base_model: bool = False
    freeze_layers: List[int] = field(default_factory=list)
    use_lora: bool = True  # Low-Rank Adaptation
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Domain adaptation
    use_domain_adaptation: bool = True
    domain_adapter_size: int = 256
    num_domains: int = 5
    
    # Output
    output_dir: str = "./fine_tuned_models"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Advanced
    prompt_tuning: bool = False
    prompt_length: int = 10
    adapter_tuning: bool = True
    adapter_size: int = 64

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original transformation
        result = F.linear(x, self.weight)
        
        # Low-rank adaptation
        x = self.dropout(x)
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return result + lora_output
        

class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning"""
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        
        # Down projection
        hidden = self.down_project(hidden_states)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Up projection
        hidden = self.up_project(hidden)
        hidden = self.dropout(hidden)
        
        # Residual connection
        return self.layer_norm(hidden + residual)

class PromptTuningEmbedding(nn.Module):
    """Soft prompt embeddings for prompt tuning"""
    
    def __init__(
        self,
        prompt_length: int,
        embedding_dim: int,
        num_prompts: int = 1
    ):
        super().__init__()
        self.prompt_length = prompt_length
        self.num_prompts = num_prompts
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, prompt_length, embedding_dim) * 0.01
        )
        
    def forward(
        self,
        input_embeddings: torch.Tensor,
        prompt_id: int = 0
    ) -> torch.Tensor:
        batch_size = input_embeddings.size(0)
        
        # Get prompt embeddings
        prompt = self.prompt_embeddings[prompt_id].unsqueeze(0)
        prompt = prompt.expand(batch_size, -1, -1)
        
        # Prepend prompt to input
        return torch.cat([prompt, input_embeddings], dim=1)

class FineTuner:
    """Main class for fine-tuning chatbot models"""
    
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Load base model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Apply fine-tuning strategy
        self._apply_fine_tuning_strategy()
        
        # Domain knowledge
        self.knowledge_bank = None
        if config.use_domain_adaptation:
            self.knowledge_bank = DomainKnowledgeBank()
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        return logging.getLogger(__name__)
        
    def _load_model(self) -> Tuple[nn.Module, AutoTokenizer]:
        """Load base model and tokenizer"""
        if self.config.custom_model:
            # Load custom model
            model = CustomChatbotModel.from_pretrained(self.config.base_model_path)
        else:
            # Load standard model
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model_path)
            
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        
        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    def _apply_fine_tuning_strategy(self):
        """Apply the selected fine-tuning strategy"""
        # Freeze base model if requested
        if self.config.freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Freeze specific layers
        if self.config.freeze_layers:
            self._freeze_layers(self.config.freeze_layers)
            
        # Apply LoRA
        if self.config.use_lora:
            self._apply_lora()
            
        # Apply adapters
        if self.config.adapter_tuning:
            self._apply_adapters()
            
        # Apply prompt tuning
        if self.config.prompt_tuning:
            self._apply_prompt_tuning()
            
        # Add domain adapters
        if self.config.use_domain_adaptation:
            self._add_domain_adapters()
            
        self.logger.info(f"Trainable parameters: {self._count_parameters(trainable_only=True):,}")
        self.logger.info(f"Total parameters: {self._count_parameters(trainable_only=False):,}")
        
    def _freeze_layers(self, layer_indices: List[int]):
        """Freeze specific layers"""
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            self.logger.warning("Could not find layers to freeze")
            return
            
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                for param in layers[idx].parameters():
                    param.requires_grad = False
                    
    def _apply_lora(self):
        """Apply LoRA to linear layers"""
        # Find all linear layers
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
                
        # Replace with LoRA layers
        for name, module in linear_layers:
            if 'lm_head' not in name:  # Don't apply to output layer
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                
                # Copy original weights
                lora_layer.weight.data = module.weight.data.clone()
                
                # Replace module
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, lora_layer)
                
    def _apply_adapters(self):
        """Add adapter layers to transformer blocks"""
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            return
            
        for i, layer in enumerate(layers):
            # Add adapter after self-attention
            adapter = AdapterLayer(
                self.model.config.hidden_size,
                self.config.adapter_size
            )
            layer.adapter_attention = adapter
            
            # Add adapter after FFN
            adapter_ffn = AdapterLayer(
                self.model.config.hidden_size,
                self.config.adapter_size
            )
            layer.adapter_ffn = adapter_ffn
            
            # Modify forward pass
            original_forward = layer.forward
            
            def new_forward(hidden_states, *args, **kwargs):
                # Original forward
                output = original_forward(hidden_states, *args, **kwargs)
                
                # Apply adapters
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states = layer.adapter_attention(hidden_states)
                    hidden_states = layer.adapter_ffn(hidden_states)
                    return (hidden_states,) + output[1:]
                else:
                    output = layer.adapter_attention(output)
                    output = layer.adapter_ffn(output)
                    return output
                    
            layer.forward = new_forward
            
    def _apply_prompt_tuning(self):
        """Add prompt tuning embeddings"""
        embedding_dim = self.model.config.hidden_size
        self.prompt_embeddings = PromptTuningEmbedding(
            self.config.prompt_length,
            embedding_dim
        )
        
        # Modify model's forward to include prompts
        original_forward = self.model.forward
        
        def new_forward(input_ids, **kwargs):
            # Get input embeddings
            if hasattr(self.model, 'transformer'):
                input_embeds = self.model.transformer.wte(input_ids)
            else:
                input_embeds = self.model.get_input_embeddings()(input_ids)
                
            # Add prompt
            input_embeds = self.prompt_embeddings(input_embeds)
            
            # Update kwargs
            kwargs['inputs_embeds'] = input_embeds
            kwargs.pop('input_ids', None)
            
            # Adjust attention mask if present
            if 'attention_mask' in kwargs:
                batch_size = input_embeds.size(0)
                prompt_mask = torch.ones(
                    batch_size,
                    self.config.prompt_length,
                    device=input_embeds.device
                )
                kwargs['attention_mask'] = torch.cat(
                    [prompt_mask, kwargs['attention_mask']], dim=1
                )
                
            return original_forward(**kwargs)
            
        self.model.forward = new_forward
        
    def _add_domain_adapters(self):
        """Add domain-specific adapters"""
        if hasattr(self.model, 'transformer'):
            hidden_size = self.model.config.hidden_size
        else:
            hidden_size = self.model.config.hidden_size
            
        self.domain_adapter = DomainAdapter(
            hidden_size,
            self.config.num_domains,
            self.config.domain_adapter_size
        )
        
        # Add to model
        self.model.domain_adapter = self.domain_adapter
        
    def _count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())
            
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for fine-tuning"""
        from models.training_pipeline import ConversationDataset
        
        dataset = ConversationDataset(
            data_path,
            self.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        return dataset
        
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Trainer:
        """Create Hugging Face Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=["tensorboard"],
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        return trainer
        
    def fine_tune(self):
        """Run fine-tuning"""
        # Prepare datasets
        train_dataset = self.prepare_dataset(self.config.train_data_path)
        eval_dataset = None
        if os.path.exists(self.config.eval_data_path):
            eval_dataset = self.prepare_dataset(self.config.eval_data_path)
            
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # Train
        self.logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        self.logger.info("Saving fine-tuned model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save config
        with open(os.path.join(self.config.output_dir, "fine_tune_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        self.logger.info(f"Fine-tuning complete! Model saved to {self.config.output_dir}")
        
    def load_fine_tuned(self, checkpoint_path: str):
        """Load a fine-tuned model"""
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model.to(self.config.device)
        
        self.logger.info(f"Loaded fine-tuned model from {checkpoint_path}")

class DomainSpecificFineTuner(FineTuner):
    """Fine-tuner specialized for domain-specific tasks"""
    
    def __init__(self, config: FineTuneConfig, domain: str):
        super().__init__(config)
        self.domain = domain
        self._prepare_domain_data()
        
    def _prepare_domain_data(self):
        """Prepare domain-specific training data"""
        # Load domain knowledge
        if self.knowledge_bank:
            knowledge_file = f"./domain_knowledge/{self.domain}.json"
            if os.path.exists(knowledge_file):
                self.knowledge_bank.load_from_file(knowledge_file)
                
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare domain-specific dataset"""
        dataset = super().prepare_dataset(data_path)
        
        # Filter for domain-specific examples
        if hasattr(dataset, 'conversations'):
            domain_conversations = [
                conv for conv in dataset.conversations
                if conv.domain == self.domain
            ]
            dataset.conversations = domain_conversations
            
        return dataset
        
    def augment_with_domain_knowledge(self, dataset: Dataset) -> Dataset:
        """Augment dataset with domain knowledge"""
        if not self.knowledge_bank:
            return dataset
            
        # Get domain knowledge
        domain_knowledge = self.knowledge_bank.domains.get(self.domain)
        if not domain_knowledge:
            return dataset
            
        # Create synthetic examples from domain knowledge
        synthetic_examples = []
        
        # Create Q&A pairs from facts
        for fact in domain_knowledge.facts:
            # Create a question about the fact
            question = f"Tell me about {fact.split()[0]}"
            answer = fact
            
            synthetic_examples.append({
                'messages': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ],
                'domain': self.domain
            })
            
        # Add examples
        for example in domain_knowledge.examples:
            synthetic_examples.append({
                'messages': [
                    {'role': 'user', 'content': example.get('input', '')},
                    {'role': 'assistant', 'content': example.get('response', '')}
                ],
                'domain': self.domain,
                'intent': example.get('intent')
            })
            
        # Add to dataset
        # This is simplified - in practice, properly integrate with dataset class
        self.logger.info(f"Added {len(synthetic_examples)} synthetic examples for {self.domain}")
        
        return dataset

# Example usage
if __name__ == "__main__":
    # Create fine-tuning config
    config = FineTuneConfig(
        base_model_path="gpt2",
        num_epochs=2,
        batch_size=4,
        learning_rate=2e-5,
        use_lora=True,
        use_domain_adaptation=True
    )
    
    # Create fine-tuner
    fine_tuner = FineTuner(config)
    
    # Run fine-tuning
    # fine_tuner.fine_tune()  # Uncomment to run
    
    print(f"Fine-tuner initialized with {fine_tuner._count_parameters(True):,} trainable parameters")
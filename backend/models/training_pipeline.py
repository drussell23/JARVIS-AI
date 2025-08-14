import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
import logging
from datetime import datetime
import shutil
from collections import defaultdict
import random

from models.custom_model import CustomChatbotModel, CustomChatbotConfig


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model config
    model_name: str = "custom-chatbot-base"
    model_size: str = "base"  # small, base, large
    
    # Data config
    train_data_path: str = "./data/train.jsonl"
    eval_data_path: str = "./data/eval.jsonl"
    max_seq_length: int = 512
    
    # Training config
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    use_amp: bool = True  # Automatic Mixed Precision
    use_gradient_checkpointing: bool = True
    use_ddp: bool = False  # Distributed Data Parallel
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Paths
    output_dir: str = "./models/custom_chatbot"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Advanced features
    use_domain_adaptation: bool = True
    use_curriculum_learning: bool = True
    use_data_augmentation: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "custom-chatbot"
    wandb_entity: Optional[str] = None
    
    # Random seed
    seed: int = 42


@dataclass
class ConversationData:
    """Data structure for conversation"""
    conversation_id: str
    messages: List[Dict[str, str]]
    domain: Optional[str] = None
    intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationDataset(Dataset):
    """Dataset for conversation data"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        use_augmentation: bool = False
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        
        # Load data
        self.conversations = self._load_data()
        
        # Domain and intent mappings
        self.domain_to_id = self._create_mapping([c.domain for c in self.conversations if c.domain])
        self.intent_to_id = self._create_mapping([c.intent for c in self.conversations if c.intent])
        
    def _load_data(self) -> List[ConversationData]:
        """Load conversation data from file"""
        conversations = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                conv = ConversationData(
                    conversation_id=data.get('id', ''),
                    messages=data['messages'],
                    domain=data.get('domain'),
                    intent=data.get('intent'),
                    metadata=data.get('metadata', {})
                )
                conversations.append(conv)
                
        return conversations
        
    def _create_mapping(self, items: List[str]) -> Dict[str, int]:
        """Create string to ID mapping"""
        unique_items = list(set(filter(None, items)))
        return {item: idx for idx, item in enumerate(unique_items)}
        
    def __len__(self) -> int:
        return len(self.conversations)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        
        # Format conversation as text
        text = self._format_conversation(conversation.messages)
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            text = self._augment_text(text)
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare output
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
        
        # Add domain and intent if available
        if conversation.domain:
            item['domain_id'] = torch.tensor(self.domain_to_id.get(conversation.domain, 0))
        if conversation.intent:
            item['intent_id'] = torch.tensor(self.intent_to_id.get(conversation.intent, 0))
            
        return item
        
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into training text"""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
        
    def _augment_text(self, text: str) -> str:
        """Apply data augmentation techniques"""
        augmentations = [
            self._paraphrase,
            self._add_typos,
            self._synonym_replacement,
        ]
        
        # Randomly apply an augmentation
        if random.random() < 0.3:
            augmentation = random.choice(augmentations)
            text = augmentation(text)
            
        return text
        
    def _paraphrase(self, text: str) -> str:
        """Simple paraphrasing (placeholder)"""
        # In practice, use a paraphrasing model
        return text
        
    def _add_typos(self, text: str) -> str:
        """Add realistic typos"""
        # Simple implementation
        if len(text) > 10 and random.random() < 0.1:
            pos = random.randint(0, len(text) - 1)
            text = text[:pos] + text[pos+1:]
        return text
        
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms (placeholder)"""
        # In practice, use WordNet or similar
        return text


class ModelTrainer:
    """Trainer for custom chatbot model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.set_seed()
        
        # Initialize model
        self.model = self._create_model()
        self.tokenizer = self._create_tokenizer()
        
        # Move model to device
        self.model.to(config.device)
        
        # Setup distributed training if enabled
        if config.use_ddp:
            self.setup_distributed()
            
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Setup experiment tracking
        if config.use_wandb:
            self.setup_wandb()
            
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config.log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def set_seed(self):
        """Set random seed for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            
    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend='nccl')
        self.model = DDP(self.model)
        
    def setup_wandb(self):
        """Setup Weights & Biases tracking"""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=self.config.__dict__,
            name=f"{self.config.model_name}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        wandb.watch(self.model)
        
    def _create_model(self) -> CustomChatbotModel:
        """Create model based on config"""
        if self.config.model_size == "small":
            from models.custom_model import ModelBuilder
            return ModelBuilder.create_small_model()
        elif self.config.model_size == "base":
            from models.custom_model import ModelBuilder
            return ModelBuilder.create_base_model()
        elif self.config.model_size == "large":
            from models.custom_model import ModelBuilder
            return ModelBuilder.create_large_model()
        else:
            raise ValueError(f"Unknown model size: {self.config.model_size}")
            
    def _create_tokenizer(self) -> AutoTokenizer:
        """Create tokenizer"""
        # Use GPT-2 tokenizer as base
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and eval dataloaders"""
        # Create datasets
        train_dataset = ConversationDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            use_augmentation=self.config.use_data_augmentation
        )
        
        eval_dataset = ConversationDataset(
            self.config.eval_data_path,
            self.tokenizer,
            self.config.max_seq_length,
            use_augmentation=False
        )
        
        # Create samplers
        train_sampler = DistributedSampler(train_dataset) if self.config.use_ddp else None
        eval_sampler = DistributedSampler(eval_dataset) if self.config.use_ddp else None
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_dataloader, eval_dataloader
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and learning rate scheduler"""
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def train(self):
        """Main training loop"""
        # Create dataloaders
        train_dataloader, eval_dataloader = self.create_dataloaders()
        
        # Calculate total steps
        num_training_steps = len(train_dataloader) * self.config.num_epochs
        
        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps)
        
        # Training loop
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Total training steps: {num_training_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Evaluate
            eval_loss = self.evaluate(eval_dataloader)
            
            # Log metrics
            self.logger.info(f"Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "learning_rate": self.scheduler.get_last_lr()[0]
                })
                
            # Save best model
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.save_model("best_model")
                
        # Save final model
        self.save_model("final_model")
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    progress_bar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "train_loss": loss.item() * self.config.gradient_accumulation_steps,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "global_step": self.global_step
                        })
                        
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate(dataloader)
                    self.logger.info(f"Step {self.global_step}: eval_loss = {eval_loss:.4f}")
                    self.model.train()
                    
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                    
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
        
    def save_model(self, name: str):
        """Save model"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        with open(os.path.join(save_path, "training_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        self.logger.info(f"Model saved to {save_path}")
        
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        self.save_model(checkpoint_path)
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(checkpoint_path, "training_state.pt"))
        
        self.logger.info(f"Checkpoint saved at step {self.global_step}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        # Load model
        self.model = CustomChatbotModel.from_pretrained(checkpoint_path)
        self.model.to(self.config.device)
        
        # Load training state
        training_state = torch.load(os.path.join(checkpoint_path, "training_state.pt"))
        self.global_step = training_state["global_step"]
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        self.best_eval_loss = training_state["best_eval_loss"]
        
        self.logger.info(f"Checkpoint loaded from step {self.global_step}")


class CurriculumLearningScheduler:
    """Implements curriculum learning for gradual difficulty increase"""
    
    def __init__(self, num_stages: int = 3):
        self.num_stages = num_stages
        self.current_stage = 0
        self.stage_thresholds = [0.8, 0.9, 0.95]  # Performance thresholds
        
    def should_advance(self, performance_metric: float) -> bool:
        """Check if we should advance to next stage"""
        if self.current_stage < self.num_stages - 1:
            if performance_metric > self.stage_thresholds[self.current_stage]:
                self.current_stage += 1
                return True
        return False
        
    def get_difficulty_weight(self) -> float:
        """Get current difficulty weight"""
        return (self.current_stage + 1) / self.num_stages


# Example usage
if __name__ == "__main__":
    # Create training config
    config = TrainingConfig(
        model_size="small",
        batch_size=8,
        num_epochs=2,
        learning_rate=5e-5,
        use_wandb=False
    )
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Start training
    # trainer.train()  # Uncomment to run training
    
    print("Training pipeline ready!")
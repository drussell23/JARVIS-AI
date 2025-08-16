"""
LangChain-powered Chatbot with M1 Optimization
Integrates LangChain's powerful features with our memory management system
"""

import logging
from typing import List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
import asyncio
import os

try:
    # LangChain imports
    from langchain_community.llms import LlamaCpp
    from langchain.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
    from langchain.chains import ConversationChain, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.agents import initialize_agent, AgentType, Tool
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains.summarize import load_summarize_chain
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

try:
    from .intelligent_chatbot import IntelligentChatbot
    from ..memory.memory_manager import M1MemoryManager, ComponentPriority
    from ..utils.intelligent_cache import IntelligentCache
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from chatbots.intelligent_chatbot import IntelligentChatbot
    from memory.memory_manager import M1MemoryManager, ComponentPriority
    from utils.intelligent_cache import IntelligentCache

logger = logging.getLogger(__name__)


class CalculatorTool:
    """Safe calculator tool for mathematical operations"""
    
    def __init__(self):
        self.name = "Calculator"
        self.description = "Useful for mathematical calculations. Input should be a mathematical expression."
    
    def _run(self, query: str) -> str:
        """Execute calculation"""
        try:
            # Clean up the query for calculation
            cleaned_query = query.strip()
            
            # Handle word-based math operations
            word_replacements = {
                " plus ": " + ",
                " minus ": " - ",
                " times ": " * ",
                " multiplied by ": " * ",
                " divided by ": " / ",
                " to the power of ": " ** ",
                " squared": " ** 2",
                " cubed": " ** 3",
                "what is ": "",
                "what's ": "",
                "calculate ": "",
                "compute ": "",
                "?": ""
            }
            
            for word, symbol in word_replacements.items():
                cleaned_query = cleaned_query.lower().replace(word, symbol)
            
            # Remove any remaining non-math characters
            cleaned_query = cleaned_query.strip()
            
            # Safe evaluation with numexpr
            import numexpr as ne
            result = ne.evaluate(cleaned_query)
            
            # Format the result nicely
            if isinstance(result, float) and result.is_integer():
                result = int(result)
                
            return f"{result}"
            
        except Exception as e:
            try:
                # Fallback to basic eval with safety
                allowed_names = {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow, "sqrt": lambda x: x**0.5
                }
                result = eval(cleaned_query, {"__builtins__": {}}, allowed_names)
                
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                    
                return f"{result}"
            except:
                return f"I couldn't calculate '{query}'. Please ensure it's a valid mathematical expression."
    
    async def _arun(self, query: str) -> str:
        """Async execution"""
        return self._run(query)


class M1OptimizedLangChainConfig:
    """Optimal LangChain configuration for M1 Macs"""
    
    # Model configurations for different memory states
    CONFIGS = {
        "minimal": {
            "model_path": os.path.expanduser("~/Documents/ai-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
            "n_ctx": 512,
            "n_batch": 128,
            "n_gpu_layers": 0,  # CPU only for minimal
            "max_tokens": 128,
            "temperature": 0.7,
            "n_threads": 4
        },
        "balanced": {
            "model_path": os.path.expanduser("~/Documents/ai-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
            "n_ctx": 2048,
            "n_batch": 512,
            "n_gpu_layers": 1,  # Partial GPU
            "max_tokens": 256,
            "temperature": 0.7,
            "n_threads": 8
        },
        "performance": {
            "model_path": os.path.expanduser("~/Documents/ai-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"),
            "n_ctx": 4096,
            "n_batch": 1024,
            "n_gpu_layers": 2,  # More GPU layers
            "max_tokens": 512,
            "temperature": 0.7,
            "n_threads": 10
        }
    }
    
    @classmethod
    def get_config(cls, memory_usage: float) -> Dict[str, Any]:
        """Get optimal config based on memory usage"""
        if memory_usage > 0.8:
            return cls.CONFIGS["minimal"]
        elif memory_usage > 0.6:
            return cls.CONFIGS["balanced"]
        else:
            return cls.CONFIGS["performance"]


class LangChainChatbot(IntelligentChatbot):
    """
    Enhanced chatbot powered by LangChain
    Provides advanced reasoning, tool use, and chain capabilities
    """
    
    def __init__(
        self,
        memory_manager: M1MemoryManager,
        max_history_length: int = 10,
        model_name: str = "langchain-chat",
        use_local_models: bool = True
    ):
        # Initialize parent
        super().__init__(memory_manager, max_history_length, model_name)
        
        self.use_local_models = use_local_models
        self.langchain_components = {}
        self.tools = []
        self.agent = None
        self.qa_chain = None
        self.conversation_chain = None
        
        # LangChain-specific cache
        self._langchain_cache = IntelligentCache(
            memory_manager=memory_manager,
            base_cache_size_mb=100,
            max_cache_size_mb=500
        )
        
        # Register LangChain components with memory manager
        self._register_langchain_components()
        
        # Initialize if memory permits
        asyncio.create_task(self._lazy_init_langchain())
    
    def _register_langchain_components(self):
        """Register LangChain components with memory manager"""
        components = [
            ("langchain_llm", ComponentPriority.HIGH, 2000),
            ("langchain_embeddings", ComponentPriority.MEDIUM, 500),
            ("langchain_vectorstore", ComponentPriority.MEDIUM, 1000),
            ("langchain_tools", ComponentPriority.LOW, 200),
        ]
        
        for name, priority, memory_mb in components:
            self.memory_manager.register_component(name, priority, memory_mb)
    
    async def _lazy_init_langchain(self):
        """Lazily initialize LangChain components when memory allows"""
        # Wait a bit for system to stabilize
        await asyncio.sleep(2)
        
        # Check if we can load LangChain
        can_load, reason = await self.memory_manager.can_load_component("langchain_llm")
        if can_load:
            await self._init_langchain_components()
    
    async def _init_langchain_components(self):
        """Initialize LangChain components"""
        try:
            logger.info("Initializing LangChain components...")
            
            # Get memory status for configuration
            memory_snapshot = await self.memory_manager.get_memory_snapshot()
            config = M1OptimizedLangChainConfig.get_config(memory_snapshot.percent)
            
            # Initialize LLM
            if self.use_local_models and os.path.exists(config["model_path"]):
                # Use llama.cpp for M1 optimization
                callbacks = [StreamingStdOutCallbackHandler()] if logger.isEnabledFor(logging.DEBUG) else []
                
                self.langchain_components["llm"] = LlamaCpp(
                    model_path=config["model_path"],
                    n_ctx=config["n_ctx"],
                    n_batch=config["n_batch"],
                    n_gpu_layers=config["n_gpu_layers"],
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    n_threads=config["n_threads"],
                    f16_kv=True,  # Use half precision
                    use_mlock=True,  # Lock model in memory
                    callbacks=callbacks,
                    verbose=False
                )
            else:
                # Fallback to HuggingFace pipeline
                from transformers import pipeline
                pipe = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    device="mps" if self.memory_manager.is_m1 else "cpu",
                    max_length=config["max_tokens"]
                )
                self.langchain_components["llm"] = HuggingFacePipeline(pipeline=pipe)
            
            # Initialize embeddings (lightweight model)
            self.langchain_components["embeddings"] = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'mps' if self.memory_manager.is_m1 else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize conversation memory
            self.langchain_components["memory"] = ConversationSummaryBufferMemory(
                llm=self.langchain_components["llm"],
                max_token_limit=2000,
                return_messages=True
            )
            
            # Initialize tools
            await self._init_tools()
            
            # Initialize conversation chain
            self.conversation_chain = ConversationChain(
                llm=self.langchain_components["llm"],
                memory=self.langchain_components["memory"],
                verbose=False
            )
            
            # Mark components as loaded
            await self.memory_manager.load_component("langchain_llm", self.langchain_components["llm"])
            
            logger.info("LangChain components initialized successfully")
            self.features_available["langchain"] = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {e}")
            self.features_available["langchain"] = False
    
    async def _init_tools(self):
        """Initialize LangChain tools"""
        try:
            # Calculator tool
            calculator = CalculatorTool()
            self.tools.append(Tool(
                name=calculator.name,
                func=calculator._run,
                coroutine=calculator._arun,
                description=calculator.description
            ))
            
            # Search tool (DuckDuckGo - no API key needed)
            search = DuckDuckGoSearchRun()
            self.tools.append(Tool(
                name="Web Search",
                func=search.run,
                description="Search the internet for current information"
            ))
            
            # Wikipedia tool
            wikipedia_wrapper = WikipediaAPIWrapper()
            self.tools.append(Tool(
                name="Wikipedia",
                func=wikipedia_wrapper.run,
                description="Search Wikipedia for detailed information"
            ))
            
            # System info tool
            self.tools.append(Tool(
                name="System Info",
                func=self._get_system_info,
                coroutine=self._aget_system_info,
                description="Get current system and memory information"
            ))
            
            # Initialize agent with tools
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.langchain_components["llm"],
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3,  # Limit iterations for memory
                agent_kwargs={
                    "prefix": """You are JARVIS, an AI assistant. You have access to the following tools:

{tools}

When asked to perform calculations or math operations, ALWAYS use the Calculator tool.
When asked for current information, use the Web Search tool.
When asked for factual knowledge, use the Wikipedia tool.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:"""
                }
            )
            
            logger.info(f"Initialized {len(self.tools)} LangChain tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            self.tools = []
    
    def _get_system_info(self, query: str = "") -> str:
        """Get system information synchronously"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._aget_system_info(query))
        except:
            return "System information unavailable"
    
    async def _aget_system_info(self, query: str = "") -> str:
        """Get system and memory information"""
        snapshot = await self.memory_manager.get_memory_snapshot()
        
        info = f"""System Status:
- Memory Usage: {snapshot.percent * 100:.1f}%
- Available Memory: {snapshot.available / (1024**3):.1f}GB
- Chatbot Mode: {self.model_name}
- LangChain Active: {self.features_available.get('langchain', False)}
- Components Loaded: {len([c for c, i in self.memory_manager.components.items() if i.is_loaded])}
"""
        return info
    
    async def generate_response(self, user_input: str) -> str:
        """Generate response using LangChain if available, otherwise fallback"""
        # Try LangChain first if available
        if self.features_available.get("langchain") and self.agent:
            try:
                # Check for tool-triggering patterns
                user_input_lower = user_input.lower()
                
                # Enhanced mathematical pattern detection
                math_patterns = [
                    "calculate", "what is", "what's", "how much", "compute",
                    "+", "-", "*", "/", "×", "÷", "plus", "minus", "times", "divided",
                    "sum", "difference", "product", "quotient", "equals", "="
                ]
                
                # Check if it contains numbers and math operations
                has_numbers = any(char.isdigit() for char in user_input)
                has_math_pattern = any(pattern in user_input_lower for pattern in math_patterns)
                
                # Check for other tool patterns
                search_patterns = ["search", "find", "look up", "google"]
                knowledge_patterns = ["tell me about", "what is a", "who is", "explain", "define"]
                current_patterns = ["current", "latest", "today", "now", "news"]
                
                if (has_numbers and has_math_pattern) or any(trigger in user_input_lower for trigger in 
                    search_patterns + knowledge_patterns + current_patterns):
                    # Use agent for tool-based queries
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, self.agent.run, user_input
                    )
                else:
                    # Use conversation chain for general chat
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, self.conversation_chain.predict, input=user_input
                    )
                
                # Cache the response
                await self._langchain_cache.set_response(
                    user_input=user_input,
                    response=response,
                    metadata={"model": "langchain", "timestamp": datetime.now()}
                )
                
                return response
                
            except Exception as e:
                logger.warning(f"LangChain error, falling back: {e}")
        
        # Fallback to parent implementation
        return await super().generate_response(user_input)
    
    async def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced knowledge search using LangChain"""
        if not self.features_available.get("langchain"):
            return await super().search_knowledge(query, k)
        
        try:
            # Use embeddings for semantic search
            embeddings = self.langchain_components.get("embeddings")
            if embeddings and hasattr(self, "vectorstore"):
                results = self.vectorstore.similarity_search(query, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": 1.0  # Chroma doesn't return scores by default
                    }
                    for doc in results
                ]
        except Exception as e:
            logger.error(f"LangChain search error: {e}")
        
        return await super().search_knowledge(query, k)
    
    async def create_qa_chain(self):
        """Create a question-answering chain for documents"""
        if self.langchain_components.get("llm"):
            self.qa_chain = load_qa_chain(
                self.langchain_components["llm"],
                chain_type="map_reduce"  # Memory efficient for long documents
            )
    
    async def summarize_text(self, text: str) -> str:
        """Summarize text using LangChain"""
        if not self.langchain_components.get("llm"):
            return "Summarization not available"
        
        try:
            # Split text if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.create_documents([text])
            
            # Create summarization chain
            chain = load_summarize_chain(
                self.langchain_components["llm"],
                chain_type="map_reduce"
            )
            
            summary = await asyncio.get_event_loop().run_in_executor(
                None, chain.run, docs
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Unable to summarize text"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get current capabilities including LangChain features"""
        capabilities = super().get_capabilities()
        
        # Add LangChain-specific capabilities
        capabilities.update({
            "langchain": {
                "enabled": self.features_available.get("langchain", False),
                "tools": [tool.name for tool in self.tools] if self.tools else [],
                "has_agent": self.agent is not None,
                "has_qa_chain": self.qa_chain is not None,
                "llm_type": type(self.langchain_components.get("llm")).__name__ if self.langchain_components.get("llm") else None
            }
        })
        
        return capabilities
    
    async def cleanup(self):
        """Clean up LangChain resources"""
        logger.info("Cleaning up LangChain components")
        
        # Clear LangChain cache
        await self._langchain_cache.cleanup()
        
        # Unload components
        for component_name in ["langchain_llm", "langchain_embeddings", "langchain_vectorstore", "langchain_tools"]:
            await self.memory_manager.unload_component(component_name)
        
        # Clear references
        self.langchain_components.clear()
        self.tools.clear()
        self.agent = None
        self.qa_chain = None
        self.conversation_chain = None
        
        # Call parent cleanup
        await super().cleanup()


# Example usage
if __name__ == "__main__":
    async def test_langchain_chatbot():
        from memory.memory_manager import M1MemoryManager
        
        # Create memory manager
        memory_manager = M1MemoryManager()
        await memory_manager.start_monitoring()
        
        # Create LangChain chatbot
        chatbot = LangChainChatbot(memory_manager)
        
        # Wait for initialization
        await asyncio.sleep(5)
        
        # Test various queries
        test_queries = [
            "Hello! How are you?",
            "What is 2 + 2?",
            "Calculate 15 * 23 + 47",
            "Search for the latest news about AI",
            "Tell me about quantum computing",
            "What's the current system status?",
        ]
        
        for query in test_queries:
            print(f"\nUser: {query}")
            response = await chatbot.generate_response(query)
            print(f"JARVIS: {response}")
            
            # Show capabilities
            if query == test_queries[-1]:
                caps = chatbot.get_capabilities()
                print(f"\nCapabilities: {caps['langchain']}")
        
        # Cleanup
        await chatbot.cleanup()
        await memory_manager.stop_monitoring()
    
    asyncio.run(test_langchain_chatbot())
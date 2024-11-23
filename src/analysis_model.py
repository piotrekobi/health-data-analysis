from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from prompts import HealthQueryPrompts
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue, Empty
import asyncio
from typing import AsyncGenerator, Dict
from concurrent.futures import ThreadPoolExecutor
import time

class AnalysisStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.token_queue = Queue()
        self.accumulated_text = ""
        self._is_generating = False
        self._last_token_time = None
    
    def on_llm_start(self, *args, **kwargs) -> None:
        self.accumulated_text = ""
        self._is_generating = True
        self._last_token_time = None
        # Clear the queue
        while True:
            try:
                self.token_queue.get_nowait()
            except Empty:
                break
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token.startswith('***'):  # Don't yield the stop token
            return
            
        self.accumulated_text += token
        self.token_queue.put_nowait({
            'type': 'token',
            'text': token
        })
        self._last_token_time = time.time()
    
    def on_llm_end(self, *args, **kwargs) -> None:
        self._is_generating = False
        # Send the complete analysis
        if self.accumulated_text:
            self.token_queue.put_nowait({
                'type': 'complete',
                'text': self.accumulated_text.strip()
            })
        self.token_queue.put_nowait(None)  # Signal completion
    
    @property
    def is_generating(self):
        return self._is_generating or not self.token_queue.empty()
    
    @property
    def is_stalled(self):
        if self._last_token_time is None:
            return False
        return time.time() - self._last_token_time > 1.0  # Consider stalled after 1 second without tokens

class AnalysisModel:
    def __init__(self, model_path: str, model_params: dict):
        self.model_path = model_path
        self.model_params = model_params
        self.token_handler = AnalysisStreamHandler()
        self._initialize_model()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _initialize_model(self):
        """Initialize the analysis generation model"""
        callback_manager = CallbackManager([self.token_handler])

        # Set analysis-specific parameters
        model_params = {
            **self.model_params,
            'temperature': 0,
            'max_tokens': 150,
            'top_p': 0.95,
            'top_k': 40,
            'stop': ['***'],
            'repeat_penalty': 1.1
        }

        self.llm = LlamaCpp(
            model_path=str(self.model_path),
            callback_manager=callback_manager,
            **model_params
        )

        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=HealthQueryPrompts.ANALYSIS_TEMPLATE,
            verbose=False
        )

    async def _run_chain(self, params: Dict) -> None:
        """Run the chain in a separate thread."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.analysis_chain.invoke,
                params
            )
        except Exception as e:
            self.token_handler.token_queue.put_nowait({'type': 'error', 'error': str(e)})
            return None

    async def stream_analysis(self, question: str, sql_query: str, data: str) -> AsyncGenerator[Dict[str, str], None]:
        """Stream analysis generation token by token."""
        try:
            # Reset the token handler
            self.token_handler.on_llm_start()
            
            # Start analysis generation
            generation_task = asyncio.create_task(self._run_chain({
                "question": question,
                "sql_query": sql_query,
                "data": data
            }))

            # Stream tokens while generation is ongoing
            last_yield_time = time.time()
            
            while True:
                try:
                    # Try to get a token without blocking
                    token_data = self.token_handler.token_queue.get_nowait()
                    
                    if token_data is None:  # End of generation
                        break
                        
                    yield token_data
                    last_yield_time = time.time()
                    
                except Empty:
                    # If no tokens and generation is complete, break
                    if not self.token_handler.is_generating:
                        break
                    
                    # If no tokens for a while, check if generation is stalled
                    if time.time() - last_yield_time > 0.01:  # 100ms without tokens
                        await asyncio.sleep(0.001)  # Short sleep to prevent CPU spinning
                    continue
                
            # Ensure generation is complete and handle any errors
            try:
                await generation_task
            except Exception as e:
                yield {'type': 'error', 'error': str(e)}

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}
        finally:
            self.token_handler._is_generating = False

    def get_current_analysis(self) -> str:
        """Get the complete analysis from accumulated tokens."""
        return self.token_handler.accumulated_text.strip()

    def generate_analysis(self, question: str, sql_query: str, data: str) -> Dict[str, str]:
        """Generate analysis (non-streaming version)"""
        try:
            self.token_handler.on_llm_start()
            self.analysis_chain.invoke({
                "question": question,
                "sql_query": sql_query,
                "data": data
            })
            return {'text': self.token_handler.accumulated_text.strip()}
            
        except Exception as e:
            raise ValueError(f"Failed to generate analysis: {str(e)}")
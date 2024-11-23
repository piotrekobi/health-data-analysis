from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from prompts import HealthQueryPrompts
from langchain.callbacks.base import BaseCallbackHandler
import re
from typing import Dict, AsyncGenerator
from queue import Queue, Empty
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

class TokenStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.token_queue = Queue()
        self.accumulated_text = ""
        self.streaming_started = False
        self.complete_sql_query = None
        self._is_generating = False
        self._last_token_time = None
    
    def on_llm_start(self, *args, **kwargs) -> None:
        self.accumulated_text = ""
        self.streaming_started = False
        self.complete_sql_query = None
        self._is_generating = True
        self._last_token_time = None
        while True:
            try:
                self.token_queue.get_nowait()
            except Empty:
                break
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.accumulated_text += token

        clean_text = self._extract_sql_query(self.accumulated_text)
        
        if clean_text:
            if not self.streaming_started:
                self.streaming_started = True
                self.token_queue.put_nowait({
                    'type': 'token',
                    'text': clean_text
                })
            else:
                prev_clean = self._extract_sql_query(self.accumulated_text[:-len(token)])
                if prev_clean and len(clean_text) > len(prev_clean):
                    new_part = clean_text[len(prev_clean):]
                    if new_part:
                        self.token_queue.put_nowait({
                            'type': 'token',
                            'text': new_part
                        })
            
            if clean_text.strip().endswith(';'):
                self.complete_sql_query = clean_text
                self.token_queue.put_nowait({
                    'type': 'complete',
                    'text': self.complete_sql_query
                })
                self._is_generating = False
                self.token_queue.put_nowait(None)

        self._last_token_time = time.time()
    
    def _extract_sql_query(self, text: str) -> str:
        text = re.sub(r'```sql\s*|```\s*', '', text)
        sql_pattern = r'(?i)(SELECT\s+.*?)(;|\Z)'
        match = re.search(sql_pattern, text, re.DOTALL)
        
        if match:
            query = match.group(1)
            if match.group(2) == ';':
                query += ';'
            query = re.sub(r'\s+', ' ', query).strip()
            return query
        
        return ""
    
    def on_llm_end(self, *args, **kwargs) -> None:
        self._is_generating = False
        if not self.complete_sql_query and self.accumulated_text:
            clean_query = self._extract_sql_query(self.accumulated_text)
            if clean_query:
                if not clean_query.endswith(';'):
                    clean_query += ';'
                self.complete_sql_query = clean_query
                self.token_queue.put_nowait({
                    'type': 'complete',
                    'text': self.complete_sql_query
                })
        self.token_queue.put_nowait(None)
    
    @property
    def is_generating(self):
        return self._is_generating or not self.token_queue.empty()
    
    @property
    def is_stalled(self):
        if self._last_token_time is None:
            return False
        return time.time() - self._last_token_time > 1.0

def sanitize_data(data: dict) -> dict:
    """Sanitize data to ensure JSON compatibility."""
    def _sanitize_value(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if isinstance(val, np.floating) else int(val)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, (list, tuple)):
            return [_sanitize_value(x) for x in val]
        if isinstance(val, dict):
            return {k: _sanitize_value(v) for k, v in val.items()}
        return val

    return {k: _sanitize_value(v) for k, v in data.items()}

class SQLQueryModel:
    def __init__(self, model_path: str, model_params: dict, llm_params: dict):
        self.model_path = model_path
        self.model_params = model_params
        self.llm_params = llm_params
        self.token_handler = TokenStreamHandler()
        self._initialize_model()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.db = None

    def _initialize_model(self):
        callback_manager = CallbackManager([self.token_handler])

        model_params = {
            **self.model_params,
            'temperature': 0,
            'max_tokens': 500,
            'stop': [';'],
            **self.llm_params
        }

        self.llm = LlamaCpp(
            model_path=str(self.model_path),
            callback_manager=callback_manager,
            **model_params
        )

        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=HealthQueryPrompts.QUERY_TEMPLATE,
            verbose=False
        )

    async def _run_chain(self, params: Dict) -> None:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.query_chain.invoke,
                params
            )
        except Exception as e:
            self.token_handler.token_queue.put_nowait({'type': 'error', 'error': str(e)})
            return None

    async def stream_query(self, question: str) -> AsyncGenerator[Dict[str, str], None]:
        try:
            self.token_handler.on_llm_start()
            generation_task = asyncio.create_task(self._run_chain({"question": question}))
            last_yield_time = time.time()
            has_yielded_completion = False
            
            while True:
                try:
                    token_data = self.token_handler.token_queue.get_nowait()
                    
                    if token_data is None:
                        break
                    
                    if token_data['type'] == 'complete' and not has_yielded_completion:
                        has_yielded_completion = True
                        try:
                            if self.db is not None and token_data.get('text'):
                                query_result = self.db.execute_query(token_data['text'])
                                sanitized_data = [sanitize_data(record) for record in query_result.to_dict(orient='records')]
                                token_data['data'] = sanitized_data
                        except Exception as e:
                            token_data['error'] = str(e)
                    
                    yield token_data
                    last_yield_time = time.time()
                    
                except Empty:
                    if not self.token_handler.is_generating:
                        break
                    
                    if time.time() - last_yield_time > 0.01:
                        await asyncio.sleep(0.001)
                    continue
            
            try:
                await generation_task
            except Exception as e:
                yield {'type': 'error', 'error': str(e)}

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}
        finally:
            self.token_handler._is_generating = False

    def get_current_query(self) -> str:
        return self.token_handler.complete_sql_query

    def generate_query(self, question: str) -> Dict[str, str]:
        try:
            self.token_handler.on_llm_start()
            self.query_chain.invoke({"question": question})
            return {'text': self.token_handler.complete_sql_query or ''}
        except Exception as e:
            raise ValueError(f"Failed to generate query: {str(e)}")
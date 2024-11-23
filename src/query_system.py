import torch
import pandas as pd
from typing import Dict, Union, AsyncGenerator
from database_generator import HealthDataDB
from sql_query_model import SQLQueryModel
from analysis_model import AnalysisModel
import asyncio
import time
from dataclasses import dataclass

def get_optimal_gpu_layers():
    """Determine optimal number of GPU layers based on available VRAM"""
    if not torch.cuda.is_available():
        return 0
        
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * 0.8
    return min(100, int(available_memory / (100 * 1024 * 1024)))

class HealthQuerySystem:
    def __init__(self, model_path: str):
        n_gpu_layers = get_optimal_gpu_layers()
        
        model_params = {
            'n_gpu_layers': n_gpu_layers,
            'n_ctx': 4096,
            'n_batch': 512,
            'verbose': False
        }
        
        llm_params = {"top_p": 0.1, "top_k": 10, "repeat_penalty": 1.1}
        self.sql_model = SQLQueryModel(model_path, model_params, llm_params)
        self.analysis_model = AnalysisModel(model_path, model_params)
        
        self._initialize_database()
        self.sql_model.db = self.db

    def _initialize_database(self):
        self.db = HealthDataDB()
        self.db.initialize_db()

    async def stream_sql_query(self, question: str) -> AsyncGenerator[Dict, None]:
        """Stream SQL query generation and execute it."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            async for token_data in self.sql_model.stream_query(question):
                yield token_data

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def stream_analysis(self, question: str, sql_query: str, data: pd.DataFrame) -> AsyncGenerator[Dict, None]:
        """Stream analysis generation for the query results."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            async for token_data in self.analysis_model.stream_analysis(
                question,
                sql_query,
                data.to_string()
            ):
                yield token_data

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self):
        """Clean up resources."""
        self.db.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
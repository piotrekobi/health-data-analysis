import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import uvicorn
import json
from query_system import HealthQuerySystem
from error_types import HealthQueryError, HealthQueryErrorType
import asyncio
import requests.exceptions

app = FastAPI(title="Health Data Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class AnalysisRequest(BaseModel):
    question: str
    sql_query: str
    data: List[Dict[str, Any]]

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

async def stream_generator(generator):
    """Convert generator output to SSE format with custom JSON encoding."""
    try:
        async for chunk in generator:
            if isinstance(chunk, dict):
                if 'error' in chunk:
                    yield f"data: {json.dumps(chunk)}\n\n"
                    return
                    
                if 'data' in chunk and chunk['type'] == 'complete':
                    try:
                        chunk['data'] = json.loads(
                            json.dumps(chunk['data'], cls=CustomJSONEncoder)
                        )
                    except Exception as e:
                        error = HealthQueryError.format_error(
                            HealthQueryErrorType.DATA_VALIDATION,
                            f"Failed to serialize data: {str(e)}"
                        )
                        yield f"data: {json.dumps(error)}\n\n"
                        return
                        
                json_str = json.dumps(chunk, cls=CustomJSONEncoder)
                yield f"data: {json_str}\n\n"
            else:
                yield f"data: {json.dumps({'text': str(chunk), 'type': 'token'})}\n\n"
    except Exception as e:
        error = HealthQueryError.format_error(
            HealthQueryErrorType.SERVER_ERROR,
            str(e)
        )
        yield f"data: {json.dumps(error)}\n\n"

query_system = HealthQuerySystem("models/Phi-3.5-mini-instruct-Q8_0.gguf")

@app.post("/query/sql/stream")
async def stream_sql_query(request: QueryRequest):
    try:
        response = StreamingResponse(
            stream_generator(query_system.stream_sql_query(request.question)),
            media_type="text/event-stream"
        )
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        return response
    except Exception as e:
        error = HealthQueryError.format_error(
            HealthQueryErrorType.QUERY_GENERATION,
            str(e)
        )
        raise HTTPException(status_code=500, detail=error)

@app.post("/query/analysis/stream")
async def stream_analysis(request: AnalysisRequest):
    try:
        if not request.data:
            error = HealthQueryError.format_error(
                HealthQueryErrorType.DATA_VALIDATION,
                "No data provided for analysis"
            )
            raise HTTPException(status_code=400, detail=error)
            
        try:
            df = pd.DataFrame(request.data)
        except Exception as e:
            error = HealthQueryError.format_error(
                HealthQueryErrorType.DATA_VALIDATION,
                f"Invalid data format: {str(e)}"
            )
            raise HTTPException(status_code=400, detail=error)
            
        # Check data size
        data_size = len(json.dumps(request.data))
        if data_size > 1_000_000:  # 1MB limit
            error = HealthQueryError.format_error(
                HealthQueryErrorType.DATA_VALIDATION,
                "Data size exceeds maximum limit for analysis"
            )
            raise HTTPException(status_code=400, detail=error)
        
        response = StreamingResponse(
            stream_generator(query_system.stream_analysis(
                request.question,
                request.sql_query,
                df
            )),
            media_type="text/event-stream"
        )
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        return response
    except HTTPException:
        raise
    except Exception as e:
        error = HealthQueryError.format_error(
            HealthQueryErrorType.MODEL_ERROR,
            str(e)
        )
        raise HTTPException(status_code=500, detail=error)

@app.on_event("shutdown")
def shutdown_event():
    query_system.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
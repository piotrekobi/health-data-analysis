import torch
import asyncio
import time
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from query_system import HealthQuerySystem
from pathlib import Path
import json
import seaborn as sns
from datetime import datetime

@dataclass
class QueryTestCase:
    question: str

@dataclass
class QueryResult:
    question: str
    success: bool
    error: str = None
    execution_time: float = 0
    sql_query: str = None

class HealthQueryEvaluator:
    def __init__(self, model_path: str, questions_file: str):
        self.model_path = model_path
        self.system = None
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load questions from file
        with open(questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        self.test_cases = [QueryTestCase(q) for q in questions]
        print(f"Loaded {len(self.test_cases)} test cases")

    async def evaluate_single_query(self, test_case: QueryTestCase) -> QueryResult:
        start_time = time.time()
        result = QueryResult(
            question=test_case.question,
            success=False
        )
        
        try:
            complete_query = None
            query_data = None
            
            async for token_data in self.system.stream_sql_query(test_case.question):
                if token_data.get('error'):
                    result.error = token_data['error']
                    break
                    
                if token_data['type'] == 'complete':
                    complete_query = token_data['text']
                    query_data = token_data.get('data')
            
            if complete_query and query_data:
                result.sql_query = complete_query
                result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            
        result.execution_time = time.time() - start_time
        return result

    async def run_evaluation(self) -> List[QueryResult]:
        self.system = HealthQuerySystem(self.model_path)
        results = []
        
        try:
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"\nEvaluating [{i}/{len(self.test_cases)}]: {test_case.question}")
                result = await self.evaluate_single_query(test_case)
                results.append(result)
                print(f"Success: {result.success}")
                if result.error:
                    print(f"Error: {result.error}")
                
            self._generate_evaluation_report(results)
            return results
            
        finally:
            self.system.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _generate_evaluation_report(self, results: List[QueryResult]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        execution_times = pd.DataFrame({
            'question': [r.question[:50] + '...' for r in results],
            'execution_time': [r.execution_time for r in results],
            'success': [r.success for r in results]
        })
        
        # Save raw results as JSON
        results_data = []
        for result in results:
            result_dict = {
                'question': result.question,
                'success': result.success,
                'execution_time': result.execution_time,
                'error': result.error,
                'sql_query': result.sql_query
            }
            results_data.append(result_dict)
            
        json_file = self.results_dir / f'evaluation_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)

async def main():
    evaluator = HealthQueryEvaluator(
        "models/Phi-3.5-mini-instruct-Q8_0.gguf",
        "data/evaluation_questions.txt"
    )
    
    try:
        print("Starting evaluation...")
        results = await evaluator.run_evaluation()
        print("\nEvaluation complete!")
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        total = len(results)
        print(f"\nSuccess rate: {successful}/{total} ({(successful/total)*100:.1f}%)")
        
        # Print failures if any
        if successful < total:
            print("\nFailed queries:")
            for result in results:
                if not result.success:
                    print(f"\nQuestion: {result.question}")
                    print(f"Error: {result.error}")
                    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    asyncio.run(main())
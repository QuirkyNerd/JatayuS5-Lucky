import asyncio
import sys
import os
import json
from typing import List, Dict, Any

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from services.evaluation_engine import run_evaluation_detailed
from utils.logging import get_logger

logger = get_logger(__name__)

# Path to gold benchmark
DATA_DIR = os.path.join(backend_dir, "data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_expanded.json")
REPORT_PATH = os.path.join(backend_dir, "scratch", "production_eval_report.json")

async def run_production_evaluation():
    from config import settings
    settings.benchmark_mode = True
    print("==================================================================")
    print("PRODUCTION RAG EVALUATION HARNESS")
    print("==================================================================")
    print(f"Dataset: {BENCHMARK_PATH}")
    
    if not os.path.exists(BENCHMARK_PATH):
        print(f"[ERROR]: Benchmark dataset not found at {BENCHMARK_PATH}")
        return

    print("--- Starting fresh inference and evaluation run...")
    print("Note: This may take several minutes depending on LLM latency.")
    
    try:
        # Run detailed evaluation (Fresh inference + comparison + stability)
        result = await run_evaluation_detailed(BENCHMARK_PATH, mode="production")
        
        if result.get("status") == "error":
            print(f"[ERROR]: EVALUATION FAILED: {result.get('message')}")
            return

        # Display Summary
        metrics = result.get("metrics", {})
        print("\n[DONE]: EVALUATION COMPLETE")
        print("-" * 40)
        print(f"Dataset Size:  {result.get('dataset_size')} cases")
        print(f"F1 Score:      {metrics.get('f1_score', 0):.3f}")
        print(f"Precision:     {metrics.get('precision', 0):.3f}")
        print(f"Recall:        {metrics.get('recall', 0):.3f}")
        print(f"Accuracy:      {metrics.get('accuracy', 0):.3f}")
        print("-" * 40)
        print("RETRIEVAL STAGE (Research Metrics):")
        print(f"MRR@10:        {metrics.get('mrr'):.3f}")
        print(f"nDCG@10:       {metrics.get('ndcg_at_10'):.3f}")
        print("-" * 40)
        print("CLINICAL FIDELITY:")
        interpretable = result.get("interpretable_metrics", {})
        print(f"Hallucination Rate:     {interpretable.get('hallucination_rate'):.1%}")
        print(f"Specificity Preserv.:   {interpretable.get('specificity_preservation'):.1%}")
        print(f"Dangerous FP Rate:      {interpretable.get('dangerous_fp_rate'):.1%}")
        print("-" * 40)
        
        # Save Report
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVED]: Detailed report saved to: {REPORT_PATH}")
        
    except Exception as e:
        print(f"[CRITICAL ERROR]: {str(e)}")
        logger.error(f"Production evaluation harness failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_production_evaluation())

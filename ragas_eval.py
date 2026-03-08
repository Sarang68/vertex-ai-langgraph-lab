# evaluation/ragas_eval.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Is answer grounded in context? (0-1)
    answer_relevancy,       # Is answer relevant to question? (0-1)
    context_precision,      # Are retrieved chunks relevant? (0-1)
    context_recall,         # Did retrieval capture all needed info? (0-1)
    answer_correctness,     # Factual match vs ground truth (0-1)
)
from ragas.llms import LangchainLLMWrapper
from langchain_google_vertexai import ChatVertexAI
from datasets import Dataset
import pandas as pd

def run_ragas_eval(rag_pipeline, golden_dataset: list[dict]) -> pd.DataFrame:
    """
    golden_dataset: [{
        "question": str,
        "ground_truth": str,       # correct answer
        "ground_truth_contexts": [str]  # relevant doc passages
    }]
    """
    llm = LangchainLLMWrapper(ChatVertexAI(model="gemini-1.5-pro", temperature=0))

    # Run pipeline on golden questions
    eval_rows = []
    for item in golden_dataset:
        result = rag_pipeline.query(item["question"])
        eval_rows.append({
            "question": item["question"],
            "answer": result.answer,
            "contexts": [c.text for c in result.chunks],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(eval_rows)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
        llm=llm
    )
    df = scores.to_pandas()
    print(f"\n{'='*50}")
    print(f"RAGAS Evaluation Results")
    print(f"{'='*50}")
    print(f"Faithfulness:       {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy:   {df['answer_relevancy'].mean():.3f}")
    print(f"Context Precision:  {df['context_precision'].mean():.3f}")
    print(f"Context Recall:     {df['context_recall'].mean():.3f}")
    print(f"Answer Correctness: {df['answer_correctness'].mean():.3f}")
    return df

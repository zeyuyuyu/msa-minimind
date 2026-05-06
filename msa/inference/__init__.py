"""MSA inference engine for Qwen3.5-9B-MSA.

Builds the three-stage MSA inference pipeline (offline encode → router top-k →
sparse generate) on top of our trained Qwen3.5-9B-MSA model. Aligns with the
EverMind-AI/MSA reference implementation but adapted to Qwen3.5's hybrid
architecture (8 full-attention + 24 GatedDeltaNet layers).
"""

from msa.inference.memory_loader import (
    BenchmarkData,
    BenchmarkSample,
    load_benchmark,
)
from msa.inference.prompt_template import (
    DocPromptFormat,
    QueryPromptFormat,
    build_doc_prompt,
    build_query_prompt,
    parse_response,
    build_msa_train_prompt,
    parse_msa_train_response,
    calculate_ir_metrics,
    exact_match,
    text_f1,
    best_qa_metrics,
    OBJ_REF_END,
    END_OF_RETRIEVE,
)
from msa.inference.router_engine import (
    EncodedCorpus,
    load_encoded_corpus,
    score_query_against_corpus,
    topk_chunks,
    topk_unique_docs,
)
from msa.inference.sparse_generator import (
    RetrievalResult,
    SparseGenerator,
)

__all__ = [
    "BenchmarkData",
    "BenchmarkSample",
    "load_benchmark",
    "DocPromptFormat",
    "QueryPromptFormat",
    "build_doc_prompt",
    "build_query_prompt",
    "parse_response",
    "build_msa_train_prompt",
    "parse_msa_train_response",
    "calculate_ir_metrics",
    "exact_match",
    "text_f1",
    "best_qa_metrics",
    "OBJ_REF_END",
    "END_OF_RETRIEVE",
    "EncodedCorpus",
    "load_encoded_corpus",
    "score_query_against_corpus",
    "topk_chunks",
    "topk_unique_docs",
    "RetrievalResult",
    "SparseGenerator",
]

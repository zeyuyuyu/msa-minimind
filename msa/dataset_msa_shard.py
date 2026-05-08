"""
MSA CPT dataset adapter for Phase-5 sharded format.

Produces the same tensor contract as msa.dataset_msa.MSACPTDataset
(see dataset_msa.py docstring) but reads the streaming-friendly shard
format produced by sample_and_shard.py:

  shard_X/
    corpus.txt       — one passage text per line (line N = global ID N)
    samples.jsonl    — {"query":str, "pos_idx":[int], "neg_idx":int?, "ds":str}

The corpus is loaded once into memory as Corpus(list[str]); samples are
lazily iterated. For 10B-token shards (~6M passages, ~10M samples), this
fits in <50GB RAM on H200 hosts.

Drop this file into msa/dataset_msa_shard.py inside msa-minimind.
"""
from __future__ import annotations

import json
from pathlib import Path

from msa.dataset_msa import Corpus, MSACPTDataset


def load_shard(
    shard_dir: str,
    tokenizer,
    num_docs: int = 32,
    max_doc_len: int = 256,
    max_query_len: int = 384,
    max_positives_used: int = 4,
    seed: int = 42,
    include_doc_text_in_target: bool = True,
    sample_limit: int = 0,
    keep_sources: list[str] | None = None,
) -> MSACPTDataset:
    p = Path(shard_dir)
    if not p.is_dir():
        raise FileNotFoundError(shard_dir)

    corpus_path = p / "corpus.txt"
    samples_path = p / "samples.jsonl"

    print(f"[load_shard] reading {corpus_path}", flush=True)
    with open(corpus_path, encoding="utf-8") as fh:
        texts = [ln.rstrip("\n") for ln in fh]
    print(f"[load_shard]   corpus size: {len(texts):,}", flush=True)

    keep_set = set(keep_sources) if keep_sources else None
    if keep_set:
        print(f"[load_shard] keep_sources filter: {sorted(keep_set)}", flush=True)

    print(f"[load_shard] reading {samples_path}", flush=True)
    samples = []
    skipped = 0
    with open(samples_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if sample_limit and len(samples) >= sample_limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            query = obj.get("query")
            pos_idx = obj.get("pos_idx") or []
            ds = obj.get("ds", "")
            if not query or not pos_idx:
                continue
            if keep_set is not None and ds not in keep_set:
                skipped += 1
                continue
            samples.append({
                "query": query,
                "positive_ids": list(pos_idx),
                "answer": "",
            })
    print(f"[load_shard]   samples kept: {len(samples):,} (skipped {skipped:,} by source filter)", flush=True)

    corpus = Corpus(texts)
    return MSACPTDataset(
        corpus=corpus,
        samples=samples,
        tokenizer=tokenizer,
        num_docs=num_docs,
        max_doc_len=max_doc_len,
        max_query_len=max_query_len,
        max_positives_used=max_positives_used,
        seed=seed,
        include_doc_text_in_target=include_doc_text_in_target,
    )

"""
Benchmark adapters for MSA Continual Pre-training.

Each adapter converts a public QA/retrieval benchmark into a uniform stream of:

    {'query': str, 'positive_texts': list[str], 'answer': str, 'source': str}

A positive_text is the **text** of a gold-standard passage (one that would lead to the
correct answer). The caller is responsible for building a persistent global corpus id
map across all adapters; adapters do not need to know or assign IDs.

Paper reference: the 9 benchmarks in §4.1 are used for EVALUATION. Paper's CPT corpus
is a 158.95B-token deduplicated corpus whose composition is not public. We substitute
by concatenating train-split signal from the same benchmarks — this keeps training and
evaluation distributions aligned, at the cost of not being corpus-identical to the paper.

Multi-hop datasets (HotpotQA, 2Wiki, MuSiQue) are split into single-step samples per
paper §3.5:
    "each retrieval chain in the multi-hop datasets is divided into multiple training
     samples during model training. Each sample contains a single retrieval step."
"""
from __future__ import annotations
import abc
import random
from typing import Iterator, Iterable, Optional


class BenchmarkAdapter(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def iter_samples(self, max_samples: int = 0) -> Iterator[dict]:
        """Yield dicts with keys {'query', 'positive_texts', 'answer', 'source'}."""


# ---------------------------------------------------------------------------
# MS MARCO (v1.1 or v2.1)
# ---------------------------------------------------------------------------
class MSMarcoAdapter(BenchmarkAdapter):
    def __init__(self, version: str = "v2.1", split: str = "train", cache_dir: Optional[str] = None):
        self.version = version
        self.split = split
        self.cache_dir = cache_dir
        self.name = f"ms_marco_{version}"

    def iter_samples(self, max_samples: int = 0):
        from datasets import load_dataset
        ds = load_dataset("microsoft/ms_marco", self.version, split=self.split, cache_dir=self.cache_dir)
        count = 0
        for ex in ds:
            passages = ex["passages"]["passage_text"]
            is_selected = ex["passages"]["is_selected"]
            pos_texts = [p for p, sel in zip(passages, is_selected) if int(sel) == 1]
            if not pos_texts:
                continue
            answers = ex.get("answers") or []
            yield {
                "query": ex["query"],
                "positive_texts": pos_texts,
                "answer": str(answers[0]) if answers else "",
                "source": self.name,
            }
            count += 1
            if max_samples and count >= max_samples:
                return


# ---------------------------------------------------------------------------
# HotpotQA (distractor variant — contains 10 paragraphs per question; 2 are gold)
# ---------------------------------------------------------------------------
class HotpotQAAdapter(BenchmarkAdapter):
    name = "hotpot_qa"

    def __init__(self, variant: str = "distractor", split: str = "train",
                 cache_dir: Optional[str] = None, split_multi_hop: bool = True):
        self.variant = variant
        self.split = split
        self.cache_dir = cache_dir
        self.split_multi_hop = split_multi_hop       # §3.5: split each chain into single-step samples

    @staticmethod
    def _paragraph_text(title: str, sentences: list[str]) -> str:
        return title + ". " + "".join(sentences).strip()

    def iter_samples(self, max_samples: int = 0):
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", self.variant, split=self.split, cache_dir=self.cache_dir,
                          trust_remote_code=True)
        count = 0
        for ex in ds:
            # context.title: list of titles; context.sentences: list of sentence lists
            titles = ex["context"]["title"]
            sentences = ex["context"]["sentences"]
            title_to_text = {t: self._paragraph_text(t, s) for t, s in zip(titles, sentences)}

            # supporting_facts.title: list of titles (duplicates possible — one per supporting sentence)
            sup_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))  # dedup, preserve order
            pos_texts = [title_to_text[t] for t in sup_titles if t in title_to_text]
            if not pos_texts:
                continue

            if self.split_multi_hop and len(pos_texts) > 1:
                # Split the multi-hop chain into single-step samples.
                # Sample 1: query → first positive passage
                # Sample 2: query + first passage → second positive passage
                # (Paper §3.5)
                running_query = ex["question"]
                for i, pt in enumerate(pos_texts):
                    yield {
                        "query": running_query,
                        "positive_texts": [pt],
                        "answer": ex["answer"] if i == len(pos_texts) - 1 else "",
                        "source": self.name,
                    }
                    running_query = running_query + "\n[retrieved] " + pt[:200]
                    count += 1
                    if max_samples and count >= max_samples:
                        return
            else:
                yield {
                    "query": ex["question"],
                    "positive_texts": pos_texts,
                    "answer": ex.get("answer", ""),
                    "source": self.name,
                }
                count += 1
                if max_samples and count >= max_samples:
                    return


# ---------------------------------------------------------------------------
# TriviaQA (rc) — entity_pages.wiki_context is the Wikipedia evidence
# ---------------------------------------------------------------------------
class TriviaQAAdapter(BenchmarkAdapter):
    name = "trivia_qa"

    def __init__(self, config: str = "rc.nocontext", split: str = "train",
                 cache_dir: Optional[str] = None, max_passage_chars: int = 2000):
        """
        config:
          rc           — has both entity_pages (Wikipedia) and search_results (web)
          rc.nocontext — questions-only (not usable for retrieval)
        Default here switches to 'rc' for actual evidence.
        """
        self.config = "rc" if config == "rc.nocontext" else config
        self.split = split
        self.cache_dir = cache_dir
        self.max_passage_chars = max_passage_chars

    def iter_samples(self, max_samples: int = 0):
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", self.config, split=self.split, cache_dir=self.cache_dir)
        count = 0
        for ex in ds:
            wiki_ctx = ex.get("entity_pages", {}).get("wiki_context", []) or []
            # Keep only non-empty contexts.
            pos_texts = []
            for c in wiki_ctx:
                if c and len(c.strip()) > 50:
                    # Truncate very long Wikipedia pages to manageable passages.
                    pos_texts.append(c[:self.max_passage_chars].strip())
            if not pos_texts:
                continue
            ans = ex.get("answer", {}) or {}
            answer = ans.get("value") or ""
            yield {
                "query": ex["question"],
                "positive_texts": pos_texts,
                "answer": answer,
                "source": self.name,
            }
            count += 1
            if max_samples and count >= max_samples:
                return


# ---------------------------------------------------------------------------
# Natural Questions — use pre-extracted (query, positive_passage) pairs
# ---------------------------------------------------------------------------
class NaturalQuestionsAdapter(BenchmarkAdapter):
    name = "natural_questions"

    def __init__(self, split: str = "train", cache_dir: Optional[str] = None):
        self.split = split
        self.cache_dir = cache_dir

    def iter_samples(self, max_samples: int = 0):
        from datasets import load_dataset
        # This pre-processed repo gives (query, answer) where `answer` is a Wikipedia passage.
        # A simpler, denser training signal than parsing the full google-research-datasets release.
        ds = load_dataset("sentence-transformers/natural-questions", split=self.split,
                          cache_dir=self.cache_dir)
        count = 0
        for ex in ds:
            q = ex.get("query")
            a = ex.get("answer")
            if not q or not a:
                continue
            yield {
                "query": q,
                "positive_texts": [a],
                "answer": "",   # This resource doesn't carry a short-answer; we train retrieval only.
                "source": self.name,
            }
            count += 1
            if max_samples and count >= max_samples:
                return


# ---------------------------------------------------------------------------
# MuSiQue — 2/3/4-hop; split chain → single-step samples (paper §3.5)
# ---------------------------------------------------------------------------
class MuSiQueAdapter(BenchmarkAdapter):
    name = "musique"

    def __init__(self, split: str = "train", cache_dir: Optional[str] = None,
                 split_multi_hop: bool = True):
        self.split = split
        self.cache_dir = cache_dir
        self.split_multi_hop = split_multi_hop

    def iter_samples(self, max_samples: int = 0):
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split=self.split, cache_dir=self.cache_dir)
        count = 0
        for ex in ds:
            if not ex.get("answerable", True):
                continue
            paragraphs = ex.get("paragraphs") or []
            # Positive paragraphs have is_supporting=True.
            pos_paras = [p for p in paragraphs if p.get("is_supporting")]
            pos_texts = []
            for p in pos_paras:
                title = (p.get("title") or "").strip()
                body = (p.get("paragraph_text") or "").strip()
                if body:
                    pos_texts.append(f"{title}. {body}" if title else body)
            if not pos_texts:
                continue

            question = ex.get("question", "")
            answer = ex.get("answer", "")

            if self.split_multi_hop and len(pos_texts) > 1:
                running_query = question
                for i, pt in enumerate(pos_texts):
                    yield {
                        "query": running_query,
                        "positive_texts": [pt],
                        "answer": answer if i == len(pos_texts) - 1 else "",
                        "source": self.name,
                    }
                    running_query = running_query + "\n[retrieved] " + pt[:200]
                    count += 1
                    if max_samples and count >= max_samples:
                        return
            else:
                yield {
                    "query": question,
                    "positive_texts": pos_texts,
                    "answer": answer,
                    "source": self.name,
                }
                count += 1
                if max_samples and count >= max_samples:
                    return


# ---------------------------------------------------------------------------
# Registry + convenience
# ---------------------------------------------------------------------------
ADAPTERS = {
    "ms_marco_v1": lambda **kw: MSMarcoAdapter(version="v1.1", **kw),
    "ms_marco_v2": lambda **kw: MSMarcoAdapter(version="v2.1", **kw),
    "hotpot_qa":   lambda **kw: HotpotQAAdapter(**kw),
    "trivia_qa":   lambda **kw: TriviaQAAdapter(**kw),
    "nq":          lambda **kw: NaturalQuestionsAdapter(**kw),
    "musique":     lambda **kw: MuSiQueAdapter(**kw),
}


def build_multi_benchmark(
    tokenizer,
    adapter_names: Iterable[str],
    per_adapter_max: int = 0,
    num_docs: int = 64,
    max_doc_len: int = 256,
    max_query_len: int = 256,
    max_positives_used: int = 4,
    seed: int = 42,
    **kwargs,
):
    """Build one MSACPTDataset backed by a unified global corpus built from every adapter.

    Each passage that appears in multiple benchmarks gets a single global id (deduped
    by exact text match), so a passage positive for one query can naturally serve as a
    negative for queries from a different benchmark.
    """
    from msa.dataset_msa import Corpus, MSACPTDataset

    corpus_texts: list[str] = []
    text_to_gid: dict[str, int] = {}
    samples: list[dict] = []
    per_source_counts: dict[str, int] = {}

    def _intern(text: str) -> int:
        g = text_to_gid.get(text)
        if g is None:
            g = len(corpus_texts)
            text_to_gid[text] = g
            corpus_texts.append(text)
        return g

    for name in adapter_names:
        if name not in ADAPTERS:
            raise ValueError(f"unknown adapter {name!r}; known={list(ADAPTERS)}")
        adapter = ADAPTERS[name]()
        emitted = 0
        for sample in adapter.iter_samples(max_samples=per_adapter_max):
            pos_gids = [_intern(t) for t in sample["positive_texts"]]
            if not pos_gids:
                continue
            samples.append({
                "query": sample["query"],
                "positive_ids": pos_gids,
                "answer": sample.get("answer", ""),
                "source": sample.get("source", name),
            })
            emitted += 1
        per_source_counts[name] = emitted
        print(f"  [{name}] emitted {emitted} training samples; corpus now {len(corpus_texts)} passages")

    corpus = Corpus(corpus_texts)
    ds = MSACPTDataset(
        corpus=corpus, samples=samples, tokenizer=tokenizer,
        num_docs=num_docs, max_doc_len=max_doc_len, max_query_len=max_query_len,
        max_positives_used=max_positives_used, seed=seed, **kwargs,
    )
    ds.per_source_counts = per_source_counts
    return ds

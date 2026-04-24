"""
Dataset for MSA Continual Pre-training (Generative Retrieval).

Three sources are supported, in increasing fidelity to the paper:

  * ``synthetic``  — a self-contained fact corpus (no external data); useful for fast CI.
  * ``t2t_mini``   — MiniMind's general pretraining corpus (a retrieval signal is
                     manufactured from passage prefix / suffix splits).
  * ``ms_marco``   — ``microsoft/ms_marco`` v1.1, matching the MS MARCO benchmark used
                     in Section 4.1 of the MSA paper. The full pretraining corpus from
                     Sec. 3.3.1 (158.95B tokens) is not public; MS MARCO is the closest
                     paper-faithful surrogate available.

All three produce the same tensor contract. Crucially:

  * Each corpus document is assigned a **persistent global integer ID** when the corpus
    is first built. The Generative-Retrieval target is written using those global IDs
    — not per-sample slot indices — so the model genuinely has to learn a corpus-wide
    id↔content mapping rather than a position-within-batch heuristic.
  * ``|P| ≥ 1`` is honoured throughout: aux-loss positive masks, target generation,
    and sampling all handle multi-positive queries (MS MARCO's ``is_selected`` field
    can label several passages for a single query).

Tensor contract per sample:

  * doc_input_ids        [N, L_d]   N documents (mix of positives and random negatives)
  * doc_attention_mask   [N, L_d]
  * query_input_ids      [L_q]      prompt + generative-retrieval target, right-padded
  * query_attention_mask [L_q]
  * pos_doc_labels       [N]        1 for positive, 0 for negative (multi-hot)
  * labels               [L_q]      supervised tokens (prompt masked to -100)
"""

from __future__ import annotations
import json
import os
import random
import string
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


OBJ_REF_END = "<|object_ref_end|>"    # special token: id 4 in MiniMind
END_OF_RETRIEVE = "<End-of-Retrieve>"  # plain-text delimiter used in paper's Fig. 3


# --------------------------------------------------------------------------------------
# Corpus wrapper — maps global IDs to passage text
# --------------------------------------------------------------------------------------
class Corpus:
    """Persistent document-id ↔ text table shared by all samples."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def get(self, gid: int) -> str:
        return self.texts[gid]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class MSACPTDataset(Dataset):
    """Continual Pre-training dataset for MSA (Generative Retrieval).

    Arguments:
        corpus:          shared Corpus with stable global IDs.
        samples:         list of dicts with keys
                             'query'         : str
                             'positive_ids'  : list[int]   (global IDs, len ≥ 1)
                             'answer'        : str         ("" if no reference answer)
        num_docs:        total docs per sample (positives + random negatives).
                         Should be ≫ top_k for meaningful sparsity.
        max_doc_len:     per-doc token budget.
        max_query_len:   query sequence token budget (prompt + target).
        max_positives_used: cap on how many positives to include per sample.
        include_doc_text_in_target: if True, injects the positive passage text after
                         the generated ID (paper's Fig. 3 pattern).
    """

    def __init__(
        self,
        corpus: Corpus,
        samples: list[dict],
        tokenizer,
        num_docs: int = 32,
        max_doc_len: int = 96,
        max_query_len: int = 160,
        max_positives_used: int = 4,
        seed: int = 42,
        include_doc_text_in_target: bool = True,
    ):
        self.corpus = corpus
        self.samples = samples
        self.tokenizer = tokenizer
        self.num_docs = num_docs
        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len
        self.max_positives_used = max_positives_used
        self.rng_seed = seed
        self.include_doc_text_in_target = include_doc_text_in_target

        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.rng_seed * 1_000_003 + index)

    # ---- encoding helpers ------------------------------------------------------------
    def _encode_doc(self, text: str) -> tuple[list[int], list[int]]:
        ids = self.tokenizer(text, add_special_tokens=False, max_length=self.max_doc_len - 1,
                             truncation=True).input_ids
        ids = [self.bos_id] + ids
        ids = ids[: self.max_doc_len]
        if len(ids) < self.max_doc_len:
            pad = [self.pad_id] * (self.max_doc_len - len(ids))
            return ids + pad, [1] * len(ids) + [0] * len(pad)
        return ids, [1] * self.max_doc_len

    def _make_prompt_and_target(self, query_text: str, pos_gids: list[int], answer: str):
        """Build the query sequence, matching MSA paper Fig. 3 / §3.5.

        Prompt (masked from loss):
            <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n

        Target (supervised), for single-step retrieval with |P| positives:
            [gid1] [gid2] ... <|object_ref_end|>\n               # generated doc-id sequence + delimiter
            [gid1]. <text_1><|object_ref_end|>\n                 # system-appended original text (per §3.5)
            [gid2]. <text_2><|object_ref_end|>\n                 # ... one block per positive
            ...
            <End-of-Retrieve>\n                                  # end-of-retrieval marker
            {answer}<|im_end|>                                   # final answer

        Including the original text block is the paper's "Original Text" mechanism, whose
        ablation (Table 4, §4.3) shows a 37.1% avg drop when disabled — so we default to
        include_doc_text_in_target=True.
        """
        tok = self.tokenizer
        prompt = (
            f"<|im_start|>user\n{query_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Part A: doc-id sequence + delimiter (what the router-driven LM head must learn to emit).
        id_seq = " ".join(f"[{g}]" for g in pos_gids) + OBJ_REF_END + "\n"
        # Part B: original-text injection, one "[gid]. <text><|object_ref_end|>\n" block per positive.
        text_block = ""
        if self.include_doc_text_in_target:
            for g in pos_gids:
                text_block += f"[{g}]. {self.corpus.get(g)}{OBJ_REF_END}\n"
        # Part C: end-of-retrieval + final answer.
        tail = f"{END_OF_RETRIEVE}\n{answer or ''}<|im_end|>"

        target = id_seq + text_block + tail

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids

        # Truncate the prompt if it alone is longer than the budget.
        prompt_ids = prompt_ids[: max(self.max_query_len - 1, 1)]
        room = self.max_query_len - len(prompt_ids)
        target_ids = target_ids[:room]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        if len(input_ids) < self.max_query_len:
            pad = self.max_query_len - len(input_ids)
            input_ids = input_ids + [self.pad_id] * pad
            labels = labels + [-100] * pad

        attn_mask = [1] * (len(prompt_ids) + len(target_ids)) + [0] * (self.max_query_len - len(prompt_ids) - len(target_ids))
        attn_mask = attn_mask[: self.max_query_len]
        return input_ids, attn_mask, labels

    # ---- one sample ------------------------------------------------------------------
    def __getitem__(self, index):
        rng = self._rng(index)
        sample = self.samples[index]
        positives: list[int] = list(sample["positive_ids"])
        rng.shuffle(positives)
        positives = positives[: self.max_positives_used]
        k_pos = len(positives)

        # Sample random negatives from the corpus excluding the positives.
        neg_pool_size = len(self.corpus)
        neg_needed = max(0, self.num_docs - k_pos)
        pos_set = set(positives)
        # Rejection sampling — corpus is typically ≫ num_docs so collisions are rare.
        negatives: list[int] = []
        while len(negatives) < neg_needed:
            g = rng.randrange(neg_pool_size)
            if g in pos_set:
                continue
            pos_set.add(g)   # prevent duplicate negatives too
            negatives.append(g)

        # Interleave positives into random slots.
        doc_gids = list(negatives)
        slots = rng.sample(range(self.num_docs), k_pos)
        slots.sort()
        for slot, g in zip(slots, positives):
            doc_gids.insert(slot, g)
        doc_gids = doc_gids[: self.num_docs]

        # Encode docs.
        doc_ids_list, doc_mask_list = [], []
        for g in doc_gids:
            ids, m = self._encode_doc(self.corpus.get(g))
            doc_ids_list.append(ids)
            doc_mask_list.append(m)
        doc_input_ids = torch.tensor(doc_ids_list, dtype=torch.long)
        doc_attention_mask = torch.tensor(doc_mask_list, dtype=torch.long)

        # Multi-hot positive mask.
        pos_doc_labels = torch.zeros(self.num_docs, dtype=torch.long)
        pos_mask_vec = [1 if g in set(positives) else 0 for g in doc_gids]
        pos_doc_labels = torch.tensor(pos_mask_vec, dtype=torch.long)

        # Build the generation target using the global IDs.
        q_ids, q_mask, q_labels = self._make_prompt_and_target(
            sample["query"], pos_gids=positives, answer=sample.get("answer", "")
        )
        query_input_ids = torch.tensor(q_ids, dtype=torch.long)
        query_attention_mask = torch.tensor(q_mask, dtype=torch.long)
        labels = torch.tensor(q_labels, dtype=torch.long)

        return {
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "pos_doc_labels": pos_doc_labels,
            "labels": labels,
        }


def collate_msa(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def build_msmarco_dataset(
    tokenizer,
    split: str = "train",
    version: str = "v2.1",
    max_queries: int = 0,                # 0 = use the full split
    num_docs: int = 64,
    max_doc_len: int = 512,
    max_query_len: int = 512,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> MSACPTDataset:
    """Build a CPT dataset from ``microsoft/ms_marco`` (v1.1 or v2.1).

    Each MS MARCO example contains a query plus 10 candidate passages, a subset of
    which is labelled ``is_selected=1``. We de-duplicate the passages across the whole
    split to build one stable corpus, assign each passage a global integer id, and emit
    one sample per query with its selected passages as positives. ``v2.1`` is ~10×
    larger than ``v1.1`` (808K vs 82K queries) and is the default.
    """
    from datasets import load_dataset
    ds = load_dataset("microsoft/ms_marco", version, split=split, cache_dir=cache_dir)
    if max_queries and max_queries > 0 and max_queries < len(ds):
        ds = ds.select(range(max_queries))

    corpus_texts: list[str] = []
    text_to_gid: dict[str, int] = {}
    samples: list[dict] = []

    def _intern(text: str) -> int:
        g = text_to_gid.get(text)
        if g is None:
            g = len(corpus_texts)
            text_to_gid[text] = g
            corpus_texts.append(text)
        return g

    for ex in ds:
        passages = ex["passages"]["passage_text"]
        is_selected = ex["passages"]["is_selected"]
        pos_gids = []
        for p, sel in zip(passages, is_selected):
            gid = _intern(p)
            if int(sel) == 1:
                pos_gids.append(gid)
        if not pos_gids:
            # Skip queries without a labelled positive.
            continue
        ans_list = ex.get("answers") or []
        answer = ans_list[0] if ans_list else ""
        samples.append({
            "query": ex["query"],
            "positive_ids": pos_gids,
            "answer": str(answer),
        })

    corpus = Corpus(corpus_texts)
    return MSACPTDataset(
        corpus=corpus,
        samples=samples,
        tokenizer=tokenizer,
        num_docs=num_docs,
        max_doc_len=max_doc_len,
        max_query_len=max_query_len,
        seed=seed,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Synthetic fact corpus (kept for quick tests — now uses persistent global IDs)
# --------------------------------------------------------------------------------------
_RELATIONS = [
    ("capital of {entity}", "is {value}"),
    ("population of {entity}", "is {value} million"),
    ("founder of {entity}", "is {value}"),
    ("color of {entity}", "is {value}"),
    ("favorite food of {entity}", "is {value}"),
    ("birth year of {entity}", "is {value}"),
    ("height of {entity}", "is {value} meters"),
    ("author of {entity}", "is {value}"),
]
_ADJS = ["blue", "red", "green", "shining", "tall", "short", "silent", "ancient", "golden", "silver"]
_ANIMALS = ["Cat", "Dog", "Eagle", "Tiger", "Whale", "Otter", "Raven", "Fox"]


def _rand_entity(rng: random.Random) -> str:
    return f"{rng.choice(_ADJS).capitalize()}{rng.choice(_ANIMALS)}-" + \
        "".join(rng.choices(string.ascii_uppercase, k=2))


def _rand_value(rng: random.Random, kind: str) -> str:
    if "population" in kind: return str(rng.randint(1, 999))
    if "birth year" in kind: return str(rng.randint(1500, 2020))
    if "height" in kind: return str(rng.randint(1, 300))
    if "color" in kind: return rng.choice(_ADJS)
    return _rand_entity(rng)


def build_synthetic_dataset(
    tokenizer,
    n_facts: int = 4096,
    num_docs: int = 32,
    max_doc_len: int = 48,
    max_query_len: int = 96,
    seed: int = 0,
    **kwargs,
) -> MSACPTDataset:
    rng = random.Random(seed)
    texts: list[str] = []
    samples: list[dict] = []
    used_entities: set[str] = set()
    while len(samples) < n_facts:
        ent = _rand_entity(rng)
        if ent in used_entities:
            continue
        used_entities.add(ent)
        rel_tmpl, val_tmpl = rng.choice(_RELATIONS)
        rel = rel_tmpl.format(entity=ent)
        val_text = val_tmpl.format(value=_rand_value(rng, rel))
        doc_text = f"The {rel} {val_text}."
        query = f"What is the {rel}?"
        answer = val_text.replace("is ", "", 1)
        gid = len(texts)
        texts.append(doc_text)
        samples.append({"query": query, "positive_ids": [gid], "answer": answer})
    corpus = Corpus(texts)
    return MSACPTDataset(corpus=corpus, samples=samples, tokenizer=tokenizer,
                        num_docs=num_docs, max_doc_len=max_doc_len, max_query_len=max_query_len,
                        seed=seed, **kwargs)


# --------------------------------------------------------------------------------------
# t2t_mini builder — retrieval via passage prefix/suffix split, with global IDs
# --------------------------------------------------------------------------------------
def build_from_t2t_mini(
    tokenizer,
    jsonl_path: str,
    max_facts: int = 16384,
    min_chars: int = 40,
    max_chars: int = 800,
    num_docs: int = 32,
    max_doc_len: int = 96,
    max_query_len: int = 160,
    seed: int = 42,
    **kwargs,
) -> MSACPTDataset:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    texts: list[str] = []
    samples: list[dict] = []
    with path.open() as f:
        for line in f:
            if len(samples) >= max_facts:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = str(obj.get("text") or obj.get("content") or "").strip()
            if not text or len(text) < min_chars:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            split_at = max(min_chars // 2, len(text) // 3)
            window = text[max(split_at - 20, 1): split_at + 20]
            for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
                pos = window.rfind(sep)
                if pos >= 0:
                    split_at = max(split_at - 20, 1) + pos + 1
                    break
            query = text[:split_at].strip()
            answer = text[split_at:].strip()
            if len(query) < 4 or len(answer) < 4:
                continue
            gid = len(texts)
            texts.append(text)
            samples.append({"query": query, "positive_ids": [gid], "answer": answer})
    if not samples:
        raise RuntimeError(f"No facts parsed from {jsonl_path}")
    corpus = Corpus(texts)
    return MSACPTDataset(corpus=corpus, samples=samples, tokenizer=tokenizer,
                        num_docs=num_docs, max_doc_len=max_doc_len, max_query_len=max_query_len,
                        seed=seed, **kwargs)

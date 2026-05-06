"""Prompt formatting for MSA RAG inference.

Mirrors the prompt protocol that the EverMind-AI/MSA reference uses
(src/utils/tools.py::compose_input + src/msa_service.py line ~1725):

  Per-doc wrapper:
    "<|im_start|>[i]. {doc}[i]<|im_end|>"

  Question wrapper:
    "\\nPlease answer the question based on the above historical "
    "document information\\n\\n{question}\\n"
    "Please return all documents related to the question\\n"

  Expected model output (parsed by src/app/benchmark.py):
    "...[d_id_1] [d_id_2] ... The answer to the question is: {answer}<|im_end|>"

Without these markers our trained Qwen3.5-9B-MSA cannot be measured against
the official paper benchmark — both the IR metric (precision/recall/F1/IoU
on retrieved doc IDs) and the QA metric (LLM-judge of the final answer)
depend on this exact format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


_DOC_ID_RE = re.compile(r"\[(\d+)\]")
_ANSWER_PREFIX = "The answer to the question is:"
_DOC_INSTR_TAIL = "\nPlease answer the question based"


@dataclass(frozen=True)
class DocPromptFormat:
    """Tunable wrapper around a single retrieved document.

    Keeping this configurable lets us A/B with EverMind's exact wrapper
    versus a slightly more permissive form, but the default mirrors the
    reference exactly.
    """

    open_token: str = "<|im_start|>"
    close_token: str = "<|im_end|>"

    def render(self, doc_text: str, doc_id: int) -> str:
        return f"{self.open_token}[{doc_id}]. {doc_text}[{doc_id}]{self.close_token}"


@dataclass(frozen=True)
class QueryPromptFormat:
    """Wrapper around the user question."""

    instruction_head: str = (
        "\nPlease answer the question based on the above historical "
        "document information\n\n"
    )
    instruction_tail: str = "\nPlease return all documents related to the question\n"

    def render(self, question: str) -> str:
        return f"{self.instruction_head}{question}{self.instruction_tail}"


def build_doc_prompt(
    docs: Iterable[str],
    fmt: DocPromptFormat | None = None,
    start_idx: int = 0,
) -> tuple[str, list[int]]:
    """Concatenate per-doc strings into one prompt block.

    Returns (prompt_string, doc_id_list) so callers can map back from the
    tokenized stream's ``doc_ids`` to the original corpus indices.
    """
    fmt = fmt or DocPromptFormat()
    parts: list[str] = []
    ids: list[int] = []
    for offset, doc in enumerate(docs):
        idx = start_idx + offset
        parts.append(fmt.render(doc, idx))
        ids.append(idx)
    return "".join(parts), ids


def build_query_prompt(
    question: str,
    fmt: QueryPromptFormat | None = None,
) -> str:
    """Wrap a single user question with the standard instruction frame."""
    fmt = fmt or QueryPromptFormat()
    return fmt.render(question)


def parse_response(generated_text: str) -> tuple[list[int], str]:
    """Extract (retrieved_doc_ids, answer_text) from a generation.

    Mirrors the parsing that EverMind's src/app/benchmark.py performs:

      * doc IDs come from any ``[<int>]`` token in the response, deduped
        while preserving first-seen order.
      * answer text is whatever follows the literal ``The answer to the
        question is:`` marker, with a trailing ``<|im_end|>`` stripped if
        present.
    """
    cleaned = generated_text.replace("<|endoftext|>", "")
    if _DOC_INSTR_TAIL in cleaned:
        cleaned = _DOC_INSTR_TAIL + cleaned.split(_DOC_INSTR_TAIL, 1)[1]

    seen: dict[int, None] = {}
    for m in _DOC_ID_RE.finditer(cleaned):
        seen[int(m.group(1))] = None
    doc_ids = list(seen)

    if _ANSWER_PREFIX in cleaned:
        answer = cleaned.split(_ANSWER_PREFIX, 1)[1]
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>", 1)[0]
        answer = answer.strip()
    else:
        answer = ""

    return doc_ids, answer


# ===========================================================================
# MSA-train format (what our SFT-S2 ckpt was trained to produce)
# ===========================================================================
# Must match ``msa.dataset_msa.MSACPTDataset._make_prompt_and_target``.
OBJ_REF_END = "<|object_ref_end|>"
END_OF_RETRIEVE = "<End-of-Retrieve>"
IM_START_USER = "<|im_start|>user\n"
IM_START_ASSIST = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def build_msa_train_prompt(question: str) -> str:
    """Build the chat-style prompt our SFT ckpt expects.

    Returns the *prefix* that the model continues from — i.e. everything up
    to ``<|im_start|>assistant\\n`` inclusive. Generation should start
    immediately with ``[g1] [g2] ...``.
    """
    return f"{IM_START_USER}{question}{IM_END}\n{IM_START_ASSIST}"


def parse_msa_train_response(generated_text: str) -> tuple[list[int], str, dict]:
    """Parse a SFT-format generation into (pred_doc_ids, answer, debug).

    The training target has three parts:

      Part A: doc-id sequence ``[g1] [g2] ...<|object_ref_end|>\\n``
      Part B: original-text blocks ``[g]. <text><|object_ref_end|>\\n`` (one per positive)
      Part C: ``<End-of-Retrieve>\\n{answer}<|im_end|>``

    For paper-aligned IR scoring we want the doc IDs from **Part A only**
    (the model's own routing decision, not the system-appended text). We
    cut at the first ``<|object_ref_end|>`` to isolate Part A. The answer
    comes from Part C, sliced between ``<End-of-Retrieve>\\n`` and
    ``<|im_end|>``.

    Returns:
        (doc_ids, answer, debug) where ``debug`` has fields:
          * ``part_a_text``: raw text of Part A (for diagnostics)
          * ``part_c_text``: raw text of Part C
          * ``saw_eor``: whether <End-of-Retrieve> was seen
          * ``saw_im_end``: whether <|im_end|> was seen
    """
    text = generated_text

    # Part A: take everything up to first <|object_ref_end|>
    if OBJ_REF_END in text:
        part_a_text = text.split(OBJ_REF_END, 1)[0]
    else:
        # Model didn't emit the delimiter — fall back to "before <End-of-Retrieve>"
        part_a_text = text.split(END_OF_RETRIEVE, 1)[0] if END_OF_RETRIEVE in text else text

    seen: dict[int, None] = {}
    for m in _DOC_ID_RE.finditer(part_a_text):
        seen[int(m.group(1))] = None
    doc_ids = list(seen)

    # Part C: answer is between <End-of-Retrieve>\n and <|im_end|>
    saw_eor = END_OF_RETRIEVE in text
    if saw_eor:
        part_c_text = text.split(END_OF_RETRIEVE, 1)[1]
    else:
        # Fallback: try The answer to the question is: marker (EverMind style)
        if _ANSWER_PREFIX in text:
            part_c_text = text.split(_ANSWER_PREFIX, 1)[1]
        else:
            part_c_text = ""
    part_c_text = part_c_text.lstrip("\n")
    saw_im_end = IM_END in part_c_text
    answer = part_c_text.split(IM_END, 1)[0].strip() if saw_im_end else part_c_text.strip()

    return doc_ids, answer, {
        "part_a_text": part_a_text,
        "part_c_text": part_c_text,
        "saw_eor": saw_eor,
        "saw_im_end": saw_im_end,
    }


# ===========================================================================
# Generic IR metric (works for both formats)
# ===========================================================================
def calculate_ir_metrics(
    true_labels: Sequence[int], pred_labels: Sequence[int]
) -> dict[str, float]:
    """Set-based precision / recall / F1 / IoU on retrieved doc IDs.

    Identical to ``src/app/benchmark.py::calculate_ir_metrics`` from the
    reference repo. Used as the primary retrieval metric for the 11-bench
    evaluation harness.
    """
    true_set = set(int(x) for x in true_labels)
    pred_set = set(int(x) for x in pred_labels)
    if not true_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    union = true_set | pred_set
    iou = len(true_set & pred_set) / len(union) if union else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


# ===========================================================================
# QA text metrics: SQuAD-style normalized EM + token-level F1
# ===========================================================================
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_answer(s: str) -> str:
    """SQuAD-standard answer normalization: lowercase, strip articles + punct,
    collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    """1 if normalized pred exactly equals normalized gold, else 0."""
    return int(_normalize_answer(pred) == _normalize_answer(gold))


def text_f1(pred: str, gold: str) -> float:
    """Token-level F1 between pred and gold (after SQuAD normalization)."""
    p_toks = _normalize_answer(pred).split()
    g_toks = _normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common: dict[str, int] = {}
    for t in p_toks:
        common[t] = min(p_toks.count(t), g_toks.count(t))
    same = sum(min(common[t], g_toks.count(t)) for t in set(common))
    if same == 0:
        return 0.0
    p = same / len(p_toks)
    r = same / len(g_toks)
    return 2 * p * r / (p + r)


def best_qa_metrics(pred: str, golds: Sequence[str]) -> dict[str, float]:
    """Take max EM / F1 over a list of acceptable gold answers."""
    if not golds:
        return {"em": 0.0, "f1": 0.0}
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(text_f1(pred, g) for g in golds)
    return {"em": float(em), "f1": float(f1)}

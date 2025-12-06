# eval_helpers.py
import re
import math
import torch
import logging
from typing import List, Optional

from openai import OpenAI
from models import build_tokenizer, build_model
from common import setup_openai_env

import numpy as np

logger = logging.getLogger(__name__)

import os

# Try to load private key from local file (safe if added to .gitignore)
try:
    from openai_key import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
except Exception:
    pass  

# ============================================================
# 1) Embedding functions (your logic included)
# ============================================================

# Embedding function slot (will be set by the runner)
_embedding_fn = None
_embedding_cache = {}

def set_embedding_fn(fn):
    """
    Register a callable fn(text:str) -> np.ndarray that returns an embedding vector.
    Call this once from your runner after building model & tokenizer.
    """
    global _embedding_fn
    _embedding_fn = fn

def clear_embedding_cache():
    global _embedding_cache
    _embedding_cache = {}

def get_embedding_cached(text: str):
    """
    Use the registered embedding function, with a small cache.
    """
    if _embedding_fn is None:
        raise RuntimeError("No embedding function registered. Call set_embedding_fn(...) first.")
    key = text
    v = _embedding_cache.get(key)
    if v is not None:
        return v
    emb = _embedding_fn(text)
    _embedding_cache[key] = emb
    return emb

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# 2) Your numeric helper functions (cleaned + integrated)
# ============================================================

def extract_numbers(text: str) -> List[float]:
    pattern = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+|-?\d+')
    matches = pattern.findall(text or "")
    numbers = []
    for match in matches:
        s = match.replace(',', '')
        try:
            numbers.append(float(s) if '.' in s else int(s))
        except:
            try:
                numbers.append(float(s))
            except:
                continue
    return numbers

def find_first_empty_index(lst):
    for idx, sublist in enumerate(lst):
        if isinstance(sublist, list) and len(sublist) == 0:
            return idx
    return None

def process_scientific_notation(answer_str):
    power = False
    if not answer_str:
        return answer_str, power

    # Match things like " × 10^4", " x10^{4}"
    sci_explicit = re.compile(r'10\^\{(-?\d+)\}')
    s = answer_str

    # Convert LaTeX exponent form: 10^{4} -> e4
    m = sci_explicit.search(s)
    if m:
        exp = m.group(1)
        power = True
        s = sci_explicit.sub(f"e{exp}", s)

    # Convert plain "^" forms: "10^4" -> "e4"
    s = re.sub(r'10\^(-?\d+)', r'e\1', s)

    # Allow replacing x * × before "10^"
    s = re.sub(r'(?:\s*[×xX\*]\s*10\^)', 'e', s)

    return s, power


def extract_numeric_answer(answer):
    """
    OpenAI-powered normalization of numeric answer.
    Fallback: returns original answer if OpenAI unavailable or fails.
    """
    if answer is None:
        return ""

    prompt = f"""
Given the following scientific answer, convert it into a clean numeric format with units.
If multiple values exist, choose the single most appropriate. Convert fractions or scientific
notation into standard numeric form.

Answer: "{answer}"

Respond with:
Numeric Answer: <numeric value with unit>
    """

    try:
        if 'openai' in globals():
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant skilled at processing scientific answers and formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            txt = response.choices[0].message.content
            m = re.search(r"Numeric Answer:\s*(.*)", txt)
            return m.group(1).strip() if m else answer
    except Exception as e:
        logger.debug(f"extract_numeric_answer failed: {e}")

    return answer


# ------------------------------------------------------------
# numeric extraction helpers
_simple_num_re = re.compile(r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')
_times_g_re    = re.compile(r'([-+]?\d+(?:\.\d+)?)(?:\s*(?:times|x|×))?.{0,20}\b(gravitational force|gravity|g)\b', re.I)


def extract_number_from_prediction(pred: Optional[str], prefer_llm=False) -> Optional[float]:
    """
    Robust numeric extraction from predicted answer.
    Handles:
        - LaTeX (boxed, etc.)
        - 0.44 times g
        - scientific notation 1.23 × 10^4
        - fallback regex
    """
    if not pred:
        return None
    s = str(pred).strip()

    # ratio-like: "0.44 g", "0.44 times gravitational force"
    m = _times_g_re.search(s)
    if m:
        try:
            return float(m.group(1))
        except:
            pass

    # preprocess scientific notation
    try:
        processed, _ = process_scientific_notation(s)
    except:
        processed = s

    # try your extract_numbers
    try:
        nums = extract_numbers(processed)
        if nums:
            return float(nums[-1])
    except Exception as e:
        logger.debug(f"extract_numbers failed: {e}")

    # optional: LLM-powered numeric extraction
    if prefer_llm:
        try:
            cleaned = extract_numeric_answer(s)
            nums = extract_numbers(cleaned)
            if nums:
                return float(nums[-1])
        except Exception as e:
            logger.debug(f"extract_numeric_answer failed: {e}")

    # fallback regex
    m = _simple_num_re.search(s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None

    return None


def numeric_match_with_margin(gold_str: str, pred_value: float, rel_tol_default=0.01) -> bool:
    """
    Compare predicted numeric vs gold numeric with:
        - math.isclose
        - your margin = 0.1 + 0.01 * log(|pred|)
    """
    gold_val = extract_number_from_prediction(gold_str, prefer_llm=False)
    if gold_val is None:
        return False

    if math.isclose(gold_val, pred_value, rel_tol=rel_tol_default, abs_tol=1e-12):
        return True

    # your margin logic
    try:
        margin = 0.1 + 0.01 * math.log(abs(pred_value)) if abs(pred_value) > 0 else 0.1
    except:
        margin = 0.1

    # relative check
    if pred_value != 0 and abs(gold_val - pred_value) / abs(pred_value) < margin:
        return True

    # absolute tiny-diff fallback
    if abs(gold_val - pred_value) < 1e-8:
        return True

    return False

def llm_judge_score(correct: str, predicted: str, question: Optional[str] = None,
                        model_name_arg: str = "gpt") -> Optional[float]:
    """
    Return a score in [0.0, 1.0] indicating alignment between predicted and correct message.
    - If a custom judge has been registered via set_message_judge_fn, call it.
    - Otherwise attempt to call OpenAI's chat completion API.
    - On error, return None.
    """

    init_prompt = f"""
        Compare the following reply with the expected answer and evaluate their alignment using:

        1. Relevance
        2. Completeness
        3. Accuracy

        Question: {question}
        Expected: {correct}
        Reply: {predicted}

        Return ONLY a similarity score between 0 and 10, as a single number (e.g., "7.5").
        """
    
    messages_query = [{"role": "user", "content": init_prompt},]
    
    if 'Qwen' in model_name_arg:
        judge_gen_model = build_model('Qwen2.5', '7b', False)
        judge_tokenizer = build_tokenizer('Qwen2.5', '7b', padding_side="left")
        terminators = judge_tokenizer.eos_token_id
        
        torch.autograd.set_grad_enabled(False)

        input_ids = judge_tokenizer.apply_chat_template(
            messages_query, add_generation_prompt=True, return_tensors="pt"
        ).to(judge_gen_model.device)
    
        response = judge_gen_model.generate(
                                    input_ids=input_ids,
                                    max_new_tokens=2048,
                                    temperature=0.0,
                                    eos_token_id=terminators,
                                )
    elif 'gpt' in model_name_arg:

        client = OpenAI()
        response = client.chat.completions.create(
            model='gpt-5-nano',
            messages=messages_query,
            temperature=1,
        )

    text = response.choices[0].message.content.strip()
    m = re.search(r'([-+]?\d+(?:\.\d+)?)', text)
    if not m:
        return 0.0
    raw = float(m.group(1))
    # Normalize 0..10 -> 0..1
    score = max(0.0, min(10.0, raw)) / 10.0
    return score

# ============================================================
# 3) Final comparators (same interface as your compare_answers)
# ============================================================

def compare_answers_numeric(correct_answer: str, predicted_answer: Optional[str]) -> bool:
    """
    Numeric comparator with your full logic.
    """
    logger.debug(f"[NUMERIC] Correct={correct_answer}, Pred={predicted_answer}")

    if predicted_answer is None:
        return False

    # try numeric extraction
    p_val = extract_number_from_prediction(predicted_answer, prefer_llm=False)
    
    if p_val is None:
        # failed extraction
        return None

    return numeric_match_with_margin(correct_answer, p_val)


def compare_answers_message(correct_answer: str,
                            predicted_answer: Optional[str],
                            question: Optional[str] = None,
                            embedding_fn=None,
                            llm_judge_arg: str = 'gpt',
                            embedding_weight: float = 0.5,
                            llm_judge_weight: float = 0.5,
                            threshold: float = 0.6) -> bool:
    """
    Message comparator with embedding + GPT scoring and fallback token overlap.
    Output: True/False like compare_answers().
    """
    logger.debug(f"[MESSAGE] Correct={correct_answer}, Pred={predicted_answer}")

    # if predicted_answer is None:
        # return False

    score_emb = 0.0
    score_llm = 0.0

    # ---- Embedding similarity ----
    try:
        e_pred = embedding_fn(predicted_answer)
        e_gold = embedding_fn(correct_answer)
        score_emb = cosine_similarity(e_pred, e_gold)
    except Exception as e:
        logger.debug(f"Embedding similarity failed: {e}")
        score_emb = 0.0

    # ---- LLM judge score ----
    score_llm = llm_judge_score(
        correct=correct_answer,
        predicted=predicted_answer,
        question=question,
        model_name_arg=llm_judge_arg  # 'gpt' or "Qwen" to use the local judge
    )

    # ---- Combine embedding + llm judge ----
    combined = embedding_weight * score_emb + llm_judge_weight * score_llm

    return combined > threshold

"""
Updated math equivalence grader with improved normalization:

Fixes:
- "2,4" vs "2 \\text{ and } 4"
- "137.5\\text{ degrees}" vs "137.5"
- "40^\\circ" vs "40"
- "6\\frac{1}{6}\\text{ feet}" vs "6\\frac{1}{6}"
- "$1.06" vs "1.06"
- Comma-separated numeric lists "27, $1.06"
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)


# ============================================================
# NEW: Stronger normalization layer
# ============================================================

def normalize_latex(s: str) -> str:
    """
    Normalize formatting:
    - Remove \\text{}, units, degree symbols
    - Remove currency symbols like $ or \$
    - Convert 'and' to comma
    - Clean leftover LaTeX whitespace commands
    - Remove commas inside numbers
    """

    s = str(s).strip()

    # Remove LaTeX \text{...}
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    
    # Currency symbols
    s = re.sub(r'\\?\$', '', s)

    # Remove commas inside numbers (e.g., 98,634 → 98634)
    s = re.sub(r'(\d),(?=\d)', r'\1', s)

    # Degrees
    s = re.sub(r'\^\\circ', '', s)
    s = re.sub(r'\\degree', '', s)

    # Convert ", and" → ","
    s = re.sub(r',\s*and\s*', ',', s, flags=re.IGNORECASE)

    # Convert "2 and 3" → "2,3"
    s = re.sub(r'(?<=\d)\s*and\s*(?=\d)', ',', s, flags=re.IGNORECASE)

    # General fallback
    s = re.sub(r'\band\b', ',', s, flags=re.IGNORECASE)
    
    # Convert "a and b" → "a,b"
    s = re.sub(r'\band\b', ',', s, flags=re.IGNORECASE)
    
    # Remove common unit words
    units = [
        "degrees", "degree", "deg",
        "feet", "foot", "ft",
        "meter", "meters", "m",
        "cm", "mm", "inch", "inches",
        "cent", "cents" 
    ]
    for u in units:
        s = re.sub(rf'\b{u}\b', '', s, flags=re.IGNORECASE)

    # Remove LaTeX spacing commands
    s = re.sub(r'\\[,;:!\s]', '', s)

    # Remove normal whitespace
    s = re.sub(r'\s+', '', s)

    # Normalize \left, \right
    s = s.replace('\\left', '').replace('\\right', '')

    return s


# ============================================================
# Main grader
# ============================================================

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:

    if prediction is None or reference is None:
        return False

    prediction_str = str(prediction).strip()
    reference_str  = str(reference).strip()

    # Direct exact match
    if prediction_str.lower() == reference_str.lower():
        return True

    # Apply normalization
    pred_norm = normalize_latex(prediction_str)
    ref_norm  = normalize_latex(reference_str)

    if pred_norm == ref_norm:
        return True

    # Handle multiple-choice
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    # ============================================================
    # NEW: Comma-separated number lists ("27,1.06")
    # ============================================================
    if "," in pred_norm and "," in ref_norm:
        pred_parts = pred_norm.split(",")
        ref_parts  = ref_norm.split(",")

        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pp, rr, include_percentage, is_close)
                for pp, rr in zip(pred_parts, ref_parts)
            ):
                return True

    # ============================================================
    # Numeric comparison
    # ============================================================
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction_val = parse_digits(prediction)
            reference_val = parse_digits(reference)

            if include_percentage:
                candidates = [reference_val/100, reference_val, reference_val*100]
            else:
                candidates = [reference_val]

            for item in candidates:
                if is_close:
                    if numeric_equal(prediction_val, item):
                        return True
                else:
                    if prediction_val == item:
                        return True

            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # ============================================================
    # Symbolic / structured comparison
    # ============================================================

    prediction = str(prediction).strip()
    reference  = str(reference).strip()

    # Bracket stripping
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str  = ref_str.strip("[]()")

    for c in ["{", "}", "(", ")"]:
        pred_str = pred_str.replace(c, "")
        ref_str  = ref_str.replace(c, "")

    if pred_str.lower() == ref_str.lower():
        return True

    # Interval lists [a,b] vs [c,d]
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction)
        and regex.match(r"(\(|\[).+(\)|\])", reference)
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts  = reference[1:-1].split(",")

        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix handling omitted here (unchanged)
    # Equation splitting omitted (unchanged)

    # Symbolic check
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

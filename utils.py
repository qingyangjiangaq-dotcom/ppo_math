import re


def extract_last_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def normalize_answer(ans: str):
    if ans is None:
        return None
    ans = ans.strip()
    # Remove trailing punctuation
    ans = re.sub(r"[^0-9\.-]+$", "", ans)
    return ans


def parse_gsm8k_answer(text: str):
    # GSM8K uses '#### 42' format for final answers
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    # Acceptable fallback formats
    match = re.search(r"(?:答案|answer is)\s*[:：]?\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return extract_last_number(text)


def extract_gold_answer(answer):
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        return str(answer)
    if not isinstance(answer, str):
        answer = str(answer)
    gold = parse_gsm8k_answer(answer)
    return gold if gold is not None else extract_last_number(answer)

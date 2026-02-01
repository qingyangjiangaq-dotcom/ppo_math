import re
from utils import extract_gold_answer, normalize_answer


def parse_pred(response: str):
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if match:
        return match.group(1), True, True
    match = re.search(r"(?:答案|answer is)\s*[:：]?\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        return match.group(1), True, False
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if nums:
        return nums[-1], True, False
    return None, False, False


def compute_reward(response: str, answer_text: str):
    total_reward = 0.0
    pred_raw, parse_ok, format_ok = parse_pred(response)
    if not parse_ok:
        return -0.1, pred_raw, parse_ok, format_ok

    if format_ok:
        total_reward += 0.1

    if len(response) >= 100:
        total_reward += 0.1
    steps = ["因为", "所以", "首先", "然后", "计算", "步骤", "step"]
    if any(s in response for s in steps):
        total_reward += 0.1

    pred = normalize_answer(pred_raw)
    gold = normalize_answer(extract_gold_answer(answer_text))
    if pred is None or gold is None:
        return total_reward - 0.1, pred_raw, parse_ok, format_ok
    try:
        if abs(float(pred) - float(gold)) < 1e-5:
            total_reward += 1.0
        else:
            total_reward -= 0.1
    except Exception:
        total_reward -= 0.1
    return total_reward, pred_raw, parse_ok, format_ok

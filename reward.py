import re
from utils import extract_gold_answer, normalize_answer


def parse_pred(response: str):
    """严格解析预测答案，移除兜底逻辑防止随机抓取数字。"""
    # 优先级1：严格格式 #### <数字>
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if match:
        return match.group(1), True, True
    
    # 优先级2：答案标记（次严格）
    match = re.search(r"(?:答案|answer is)\s*[:：]?\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        return match.group(1), True, False
    
    # 兜底逻辑：仅在回复极短（<20字符）时启用，防止滥用
    if len(response) < 20:
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        if nums:
            return nums[-1], True, False
    
    return None, False, False


def compute_reward(response: str, answer_text: str):
    """
    严格的奖励函数：移除所有中间奖励，只有答对才有正分，答错即负分。
    防止奖励黑客（Reward Hacking）。
    """
    pred_raw, parse_ok, format_ok = parse_pred(response)
    
    # 1. 硬性惩罚：解析失败直接重罚 -1.0
    if not parse_ok:
        return -1.0, pred_raw, parse_ok, format_ok
    
    # 2. 移除长度奖励和关键词奖励！
    # 不要教模型怎么“长得像答案”，只教它“什么是对的”。
    
    pred = normalize_answer(pred_raw)
    gold = normalize_answer(extract_gold_answer(answer_text))
    
    if pred is None or gold is None:
        return -1.0, pred_raw, parse_ok, format_ok
    
    try:
        # 3. 只有做对才有分，做错就是负分
        if abs(float(pred) - float(gold)) < 1e-5:
            # 答对了：基础奖励 +1.0
            # 格式奖励：只有答对且格式正确才给额外 +0.5
            bonus = 0.5 if format_ok else 0.0
            return 1.0 + bonus, pred_raw, parse_ok, format_ok
        else:
            # 答错了：-0.5（比解析失败的 -1.0 轻一些）
            return -0.5, pred_raw, parse_ok, format_ok
    except Exception:
        # 数值比较异常，视为错误
        return -1.0, pred_raw, parse_ok, format_ok

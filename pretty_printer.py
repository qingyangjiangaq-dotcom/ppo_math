"""美观的训练监控和打印工具"""
from datetime import datetime
from typing import Dict, List, Any

class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """打印大标题"""
    width = 70
    print(f"\n{Colors.HEADER}{'='*width}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(width)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*width}{Colors.END}\n")


def print_subheader(text: str):
    """打印小标题"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {text}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.END}")


def print_metric(name: str, value: float, unit: str = "", color: str = None, width: int = 20):
    """打印单个指标"""
    if color is None:
        if value > 0.7:
            color = Colors.GREEN
        elif value > 0.3:
            color = Colors.YELLOW
        else:
            color = Colors.RED
    
    name_str = f"{name}:".ljust(width)
    value_str = f"{value:.4f}{unit}"
    print(f"  {Colors.BOLD}{name_str}{Colors.END} {color}{value_str}{Colors.END}")


def print_metrics_table(metrics: Dict[str, float], title: str = "指标"):
    """打印指标表格"""
    print_subheader(title)
    
    # 分类显示
    categories = {
        "奖励相关": ["mean_reward", "std_reward", "parse_rate", "format_rate", "accuracy"],
        "训练相关": ["policy_loss", "value_loss", "total_loss", "approx_kl", "entropy", "clip_frac"],
        "效率相关": ["step_time_sec", "gen_tokens_per_sec", "samples_per_sec", "gpu_mem_gb"]
    }
    
    for category, keys in categories.items():
        has_any = any(k in metrics for k in keys)
        if not has_any:
            continue
            
        print(f"\n{Colors.YELLOW}{category}:{Colors.END}")
        for key in keys:
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                # 格式化显示
                if "rate" in key or "accuracy" in key or "frac" in key:
                    print_metric(key.replace("_", " ").title(), value, "", width=18)
                elif "loss" in key or "kl" in key or "entropy" in key:
                    print_metric(key.replace("_", " ").title(), value, "", width=18)
                elif "time" in key:
                    print_metric(key.replace("_", " ").title(), value, "s", width=18)
                elif "mem" in key:
                    print_metric(key.replace("_", " ").title(), value, "GB", width=18)
                elif "per_sec" in key:
                    print_metric(key.replace("_", " ").title(), value, "/s", width=18)
                else:
                    print_metric(key.replace("_", " ").title(), value, "", width=18)


def print_response_analysis(response: str, pred: str, gold: str, reward: float, 
                           question: str = "", sample_idx: int = 0):
    """美观地打印模型回复和分析"""
    print_subheader(f"样本 #{sample_idx + 1}")
    
    if question:
        print(f"{Colors.YELLOW}问题:{Colors.END} {question[:80]}..." if len(question) > 80 else f"{Colors.YELLOW}问题:{Colors.END} {question}")
    
    # 判断状态
    is_correct = pred == gold
    is_parsed = pred is not None and pred != ""
    
    # 打印预测结果
    status_color = Colors.GREEN if is_correct else (Colors.YELLOW if is_parsed else Colors.RED)
    status_text = "✓ 正确" if is_correct else ("⚠ 解析成功但错误" if is_parsed else "✗ 解析失败")
    
    print(f"\n{Colors.BOLD}预测状态:{Colors.END} {status_color}{status_text}{Colors.END}")
    print(f"{Colors.BLUE}模型预测:{Colors.END} {pred if pred else 'N/A'}")
    print(f"{Colors.BLUE}标准答案:{Colors.END} {gold if gold else 'N/A'}")
    
    # 打印奖励
    reward_color = Colors.GREEN if reward > 0.5 else (Colors.YELLOW if reward > -0.3 else Colors.RED)
    print(f"{Colors.BOLD}获得奖励:{Colors.END} {reward_color}{reward:+.2f}{Colors.END}")
    
    # 打印回复内容（格式化）
    print(f"\n{Colors.CYAN}模型回复:{Colors.END}")
    print(f"{Colors.CYAN}┌{'─'*68}┐{Colors.END}")
    
    # 智能格式化回复
    lines = response.strip().split('\n')
    for i, line in enumerate(lines[:15]):  # 最多显示15行
        if len(line) > 68:
            line = line[:65] + "..."
        print(f"{Colors.CYAN}│{Colors.END} {line.ljust(66)} {Colors.CYAN}│{Colors.END}")
    
    if len(lines) > 15:
        print(f"{Colors.CYAN}│{Colors.END} ... ({len(lines)-15} 行省略)".ljust(67) + f"{Colors.CYAN}│{Colors.END}")
    
    print(f"{Colors.CYAN}└{'─'*68}┘{Colors.END}")


def analyze_training_status(metrics: Dict[str, float], step: int) -> str:
    """分析训练状态并返回简短诊断"""
    analysis = []
    
    # 奖励分析
    mean_reward = metrics.get("mean_reward", 0)
    parse_rate = metrics.get("parse_rate", 0)
    format_rate = metrics.get("format_rate", 0)
    
    if mean_reward > 0.5:
        analysis.append(f"{Colors.GREEN}奖励良好{Colors.END} (avg: {mean_reward:.2f})")
    elif mean_reward > 0:
        analysis.append(f"{Colors.YELLOW}奖励一般{Colors.END} (avg: {mean_reward:.2f})")
    else:
        analysis.append(f"{Colors.RED}奖励偏低{Colors.END} (avg: {mean_reward:.2f}) ⚠️ 可能需检查奖励函数")
    
    # 解析率分析
    if parse_rate > 0.8:
        analysis.append(f"{Colors.GREEN}解析率高{Colors.END} ({parse_rate:.1%})")
    elif parse_rate > 0.5:
        analysis.append(f"{Colors.YELLOW}解析率中等{Colors.END} ({parse_rate:.1%})")
    else:
        analysis.append(f"{Colors.RED}解析率低{Colors.END} ({parse_rate:.1%}) ⚠️ 模型未学会格式")
    
    # 格式率分析
    if format_rate > 0.8:
        analysis.append(f"{Colors.GREEN}格式正确率高{Colors.END} ({format_rate:.1%})")
    elif format_rate > 0.5:
        analysis.append(f"{Colors.YELLOW}格式率中等{Colors.END} ({format_rate:.1%})")
    else:
        analysis.append(f"{Colors.RED}格式率低{Colors.END} ({format_rate:.1%})")
    
    # KL散度分析
    kl = metrics.get("approx_kl")
    if kl is not None:
        if kl > 0.5:
            analysis.append(f"{Colors.RED}KL过高{Colors.END} ({kl:.3f}) ⚠️ 策略偏离过大")
        elif kl > 0.1:
            analysis.append(f"{Colors.GREEN}KL正常{Colors.END} ({kl:.3f})")
        else:
            analysis.append(f"{Colors.YELLOW}KL较低{Colors.END} ({kl:.3f})")
    
    # 损失分析
    policy_loss = metrics.get("policy_loss")
    if policy_loss is not None and abs(policy_loss) > 1.0:
        analysis.append(f"{Colors.RED}策略损失过大{Colors.END} ({policy_loss:.3f})")
    
    return " | ".join(analysis)


def print_training_step(step: int, metrics: Dict[str, float], 
                       responses: List[str], preds: List[str], golds: List[str],
                       rewards: List[float], questions: List[str] = None):
    """打印完整的训练步骤信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # 大标题
    print_header(f"🚀 训练步骤 {step} | {timestamp}")
    
    # 分析状态
    print_subheader("📊 状态分析")
    analysis = analyze_training_status(metrics, step)
    print(f"  {analysis}\n")
    
    # 打印指标表格
    print_metrics_table(metrics, "📈 详细指标")
    
    # 打印样本回复
    if responses:
        print_subheader("💬 模型回复样例")
        for i, (resp, pred, gold, reward) in enumerate(zip(responses[:3], preds[:3], golds[:3], rewards[:3])):
            q = questions[i] if questions and i < len(questions) else ""
            print_response_analysis(resp, pred, gold, reward, q, i)
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}\n")


if __name__ == "__main__":
    # 测试打印效果
    test_metrics = {
        "mean_reward": 0.85,
        "std_reward": 0.15,
        "parse_rate": 0.92,
        "format_rate": 0.88,
        "approx_kl": 0.15,
        "entropy": 0.42,
        "clip_frac": 0.12,
        "policy_loss": 0.03,
        "value_loss": 0.08,
        "total_loss": 0.11,
        "step_time_sec": 2.34,
        "gen_tokens_per_sec": 125.5,
        "gpu_mem_gb": 6.8
    }
    
    test_responses = [
        "首先，小明有3个苹果。\n然后，他吃掉1个。\n所以剩下 3-1=2 个。\n#### 2",
        "这是一个复杂的问题。\n让我思考一下...\n答案是 42。\n#### 42"
    ]
    
    print_training_step(
        step=10,
        metrics=test_metrics,
        responses=test_responses,
        preds=["2", "42"],
        golds=["2", "43"],
        rewards=[1.5, -0.5],
        questions=["小明有3个苹果，吃掉1个，剩几个？", "5+3*7=?"]
    )

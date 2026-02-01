#!/usr/bin/env python3
"""
美观的训练实时监控脚本
每隔10个step展示模型回复和指标分析
"""
import json
import sys
import time
import os
from datetime import datetime
from pathlib import Path

# 颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title):
    """打印大标题"""
    width = 80
    print(f"\n{Colors.HEADER}{'='*width}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(width)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*width}{Colors.END}\n")

def print_section(title):
    """打印小节标题"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {title}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}")

def print_metric(name, value, unit=""):
    """打印单个指标"""
    # 根据数值选择颜色
    if isinstance(value, (int, float)):
        if value >= 0.7:
            color = Colors.GREEN
        elif value >= 0.3:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        value_str = f"{value:.4f}{unit}"
    else:
        color = Colors.BLUE
        value_str = f"{value}{unit}"
    
    print(f"  {Colors.BOLD}{name:25}{Colors.END} {color}{value_str}{Colors.END}")

def analyze_metrics(metrics):
    """分析指标并返回诊断"""
    analysis = []
    
    mean_reward = metrics.get("mean_reward", 0)
    parse_rate = metrics.get("parse_rate", 0)
    format_rate = metrics.get("format_rate", 0)
    
    # 奖励分析
    if mean_reward > 0.5:
        analysis.append(("奖励良好", Colors.GREEN, f"avg: {mean_reward:.2f}"))
    elif mean_reward > 0:
        analysis.append(("奖励偏低", Colors.YELLOW, f"avg: {mean_reward:.2f}"))
    else:
        analysis.append(("⚠️ 奖励过低", Colors.RED, f"avg: {mean_reward:.2f} - 需检查奖励函数"))
    
    # 解析率分析
    if parse_rate > 0.8:
        analysis.append(("解析率高", Colors.GREEN, f"{parse_rate:.1%}"))
    elif parse_rate > 0.5:
        analysis.append(("解析率中等", Colors.YELLOW, f"{parse_rate:.1%}"))
    else:
        analysis.append(("⚠️ 解析率低", Colors.RED, f"{parse_rate:.1%} - 模型未学会格式"))
    
    # 格式率分析
    if format_rate > 0.8:
        analysis.append(("格式正确率高", Colors.GREEN, f"{format_rate:.1%}"))
    elif format_rate > 0.5:
        analysis.append(("格式率中等", Colors.YELLOW, f"{format_rate:.1%}"))
    else:
        analysis.append(("⚠️ 格式率低", Colors.RED, f"{format_rate:.1%}"))
    
    # KL分析
    kl = metrics.get("approx_kl")
    if kl is not None:
        if kl > 0.5:
            analysis.append(("⚠️ KL过高", Colors.RED, f"{kl:.3f} - 策略偏离过大"))
        elif kl > 0.1:
            analysis.append(("KL正常", Colors.GREEN, f"{kl:.3f}"))
        else:
            analysis.append(("KL较低", Colors.YELLOW, f"{kl:.3f}"))
    
    return analysis

def print_metrics_dashboard(metrics):
    """打印指标仪表盘"""
    print_section("📊 训练指标")
    
    # 核心指标
    print(f"\n{Colors.YELLOW}核心指标:{Colors.END}")
    core_metrics = [
        ("平均奖励", metrics.get("mean_reward", 0), ""),
        ("奖励标准差", metrics.get("std_reward", 0), ""),
        ("解析成功率", metrics.get("parse_rate", 0), ""),
        ("格式正确率", metrics.get("format_rate", 0), ""),
    ]
    for name, value, unit in core_metrics:
        print_metric(name, value, unit)
    
    # 训练指标
    print(f"\n{Colors.YELLOW}训练指标:{Colors.END}")
    train_metrics = [
        ("策略损失", metrics.get("policy_loss", 0), ""),
        ("价值损失", metrics.get("value_loss", 0), ""),
        ("总损失", metrics.get("total_loss", 0), ""),
        ("KL散度", metrics.get("approx_kl", 0), ""),
        ("熵", metrics.get("entropy", 0), ""),
        ("Clip比例", metrics.get("clip_frac", 0), ""),
    ]
    for name, value, unit in train_metrics:
        if value is not None:
            print_metric(name, value, unit)
    
    # 效率指标
    print(f"\n{Colors.YELLOW}效率指标:{Colors.END}")
    eff_metrics = [
        ("生成token/秒", metrics.get("gen_tokens_per_sec", 0), ""),
        ("样本/秒", metrics.get("samples_per_sec", 0), ""),
        ("GPU显存", metrics.get("gpu_mem_gb", 0), " GB"),
        ("回复长度", metrics.get("response_len", 0), ""),
    ]
    for name, value, unit in eff_metrics:
        if value is not None:
            print_metric(name, value, unit)

def print_sample_analysis(sample, idx):
    """美观地打印样本分析"""
    print_section(f"样本 #{idx + 1}")
    
    question = sample.get("question", "")
    response = sample.get("response", "")
    pred = sample.get("pred", "")
    gold = sample.get("gold", "")
    reward = sample.get("reward", 0)
    
    # 打印问题
    if len(question) > 80:
        question = question[:77] + "..."
    print(f"{Colors.YELLOW}问题:{Colors.END} {question}\n")
    
    # 判断状态
    is_correct = pred == gold
    is_parsed = pred is not None and pred != ""
    
    # 状态标签
    if is_correct:
        status = f"{Colors.GREEN}{Colors.BOLD}✓ 回答正确{Colors.END}"
    elif is_parsed:
        status = f"{Colors.YELLOW}{Colors.BOLD}⚠ 解析成功但错误{Colors.END}"
    else:
        status = f"{Colors.RED}{Colors.BOLD}✗ 解析失败{Colors.END}"
    
    print(f"状态: {status}")
    print(f"预测: {Colors.BLUE}{pred or 'N/A'}{Colors.END}")
    print(f"答案: {Colors.GREEN}{gold or 'N/A'}{Colors.END}")
    
    # 奖励
    reward_color = Colors.GREEN if reward > 0.5 else (Colors.YELLOW if reward > -0.3 else Colors.RED)
    print(f"奖励: {reward_color}{reward:+.2f}{Colors.END}\n")
    
    # 打印回复（格式化）
    print(f"{Colors.CYAN}模型回复:{Colors.END}")
    print(f"{Colors.CYAN}┌{'─'*78}┐{Colors.END}")
    
    lines = response.strip().split('\n')
    for i, line in enumerate(lines[:12]):  # 最多显示12行
        if len(line) > 76:
            line = line[:73] + "..."
        print(f"{Colors.CYAN}│{Colors.END} {line.ljust(76)} {Colors.CYAN}│{Colors.END}")
    
    if len(lines) > 12:
        print(f"{Colors.CYAN}│{Colors.END} ... ({len(lines)-12} 行省略)".ljust(77) + f"{Colors.CYAN}│{Colors.END}")
    
    print(f"{Colors.CYAN}└{'─'*78}┘{Colors.END}\n")

def display_training_step(step, metrics, samples):
    """显示完整的训练步骤信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # 清空屏幕（可选）
    # os.system('clear' if os.name != 'nt' else 'cls')
    
    # 大标题
    print_header(f"🚀 训练步骤 {step} | {timestamp}")
    
    # 状态分析
    print_section("🔍 状态诊断")
    analysis = analyze_metrics(metrics)
    for text, color, detail in analysis:
        print(f"  {color}● {text}: {detail}{Colors.END}")
    print()
    
    # 指标仪表盘
    print_metrics_dashboard(metrics)
    
    # 样本分析
    if samples:
        print_section("💬 模型回复样例")
        for i, sample in enumerate(samples[:3]):  # 显示前3个样本
            print_sample_analysis(sample, i)
    
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

def monitor_training():
    """监控训练过程"""
    metrics_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/metrics.jsonl")
    samples_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/samples.jsonl")
    
    if not metrics_file.exists():
        print(f"{Colors.RED}错误: 找不到 metrics.jsonl{Colors.END}")
        return
    
    print(f"{Colors.GREEN}开始监控训练...{Colors.END}")
    print(f"监控文件: {metrics_file}")
    print(f"样本文件: {samples_file}")
    print(f"{Colors.YELLOW}提示: 按 Ctrl+C 退出{Colors.END}\n")
    
    last_step = -1
    
    try:
        while True:
            # 读取最新的指标
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 获取最后一个step的指标
                        for line in reversed(lines):
                            try:
                                metrics = json.loads(line)
                                step = metrics.get("step", 0)
                                
                                # 只处理每10个step
                                if step % 10 == 0 and step != last_step:
                                    # 读取对应的样本
                                    samples = []
                                    if samples_file.exists():
                                        with open(samples_file, 'r') as sf:
                                            for sline in sf:
                                                try:
                                                    sample = json.loads(sline)
                                                    if sample.get("step") == step:
                                                        samples.append(sample)
                                                except:
                                                    pass
                                    
                                    # 显示
                                    display_training_step(step, metrics, samples)
                                    last_step = step
                                break
                            except:
                                continue
            
            time.sleep(2)  # 每2秒检查一次
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}监控已停止{Colors.END}")

def show_historical_data():
    """显示历史训练数据"""
    metrics_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/metrics.jsonl")
    samples_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/samples.jsonl")
    
    if not metrics_file.exists():
        print(f"{Colors.RED}错误: 找不到训练数据{Colors.END}")
        return
    
    # 读取所有指标
    all_metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metrics = json.loads(line)
                all_metrics.append(metrics)
            except:
                pass
    
    # 只显示每10个step的数据
    for metrics in all_metrics:
        step = metrics.get("step", 0)
        if step % 10 == 0:
            # 读取对应的样本
            samples = []
            if samples_file.exists():
                with open(samples_file, 'r') as sf:
                    for line in sf:
                        try:
                            sample = json.loads(line)
                            if sample.get("step") == step:
                                samples.append(sample)
                        except:
                            pass
            
            display_training_step(step, metrics, samples)
            
            # 询问是否继续
            if step < all_metrics[-1].get("step", 0):
                try:
                    input(f"{Colors.YELLOW}按 Enter 查看下一步，或按 Ctrl+C 退出...{Colors.END}")
                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}已退出{Colors.END}")
                    break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        show_historical_data()
    else:
        monitor_training()

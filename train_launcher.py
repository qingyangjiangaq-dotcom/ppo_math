#!/usr/bin/env python3
"""
智能训练启动器 - 自动监控并每隔10个step调用分析
"""
import subprocess
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# 颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """打印启动横幅"""
    print(f"""
{Colors.CYAN}{'='*80}{Colors.END}
{Colors.CYAN}{Colors.BOLD}{'🚀 PPO 数学训练启动器':^80}{Colors.END}
{Colors.CYAN}{'='*80}{Colors.END}

{Colors.YELLOW}功能:{Colors.END}
  • 启动训练进程
  • 实时监控训练指标
  • 每隔10个step自动分析训练状况
  • 彩色格式化输出

{Colors.YELLOW}使用方法:{Colors.END}
  python train_launcher.py [--steps 100]

{Colors.CYAN}{'='*80}{Colors.END}
""")

def analyze_training_step(step_data):
    """简单分析训练步骤"""
    metrics = step_data['metrics']
    samples = step_data['samples']
    
    analysis = []
    
    # 奖励分析
    mean_reward = metrics.get('mean_reward', 0)
    if mean_reward > 0.5:
        analysis.append(f"{Colors.GREEN}✓ 奖励优秀 ({mean_reward:.2f}){Colors.END}")
    elif mean_reward > 0:
        analysis.append(f"{Colors.YELLOW}⚠ 奖励偏低 ({mean_reward:.2f}){Colors.END}")
    else:
        analysis.append(f"{Colors.RED}✗ 奖励过低 ({mean_reward:.2f}) - 需检查{Colors.END}")
    
    # 解析率和格式率
    parse_rate = metrics.get('parse_rate', 0)
    format_rate = metrics.get('format_rate', 0)
    
    if parse_rate > 0.8:
        analysis.append(f"{Colors.GREEN}✓ 解析率高 ({parse_rate:.1%}){Colors.END}")
    else:
        analysis.append(f"{Colors.RED}✗ 解析率低 ({parse_rate:.1%}){Colors.END}")
    
    if format_rate > 0.8:
        analysis.append(f"{Colors.GREEN}✓ 格式正确 ({format_rate:.1%}){Colors.END}")
    else:
        analysis.append(f"{Colors.YELLOW}⚠ 格式率需提升 ({format_rate:.1%}){Colors.END}")
    
    # 样本质量分析
    correct_count = sum(1 for s in samples if s.get('pred') == s.get('gold'))
    total = len(samples)
    accuracy = correct_count / total if total > 0 else 0
    
    if accuracy > 0.5:
        analysis.append(f"{Colors.GREEN}✓ 样本准确率 {accuracy:.1%} ({correct_count}/{total}){Colors.END}")
    elif accuracy > 0.2:
        analysis.append(f"{Colors.YELLOW}⚠ 样本准确率 {accuracy:.1%} ({correct_count}/{total}){Colors.END}")
    else:
        analysis.append(f"{Colors.RED}✗ 样本准确率 {accuracy:.1%} ({correct_count}/{total}){Colors.END}")
    
    # 响应长度分析
    resp_len = metrics.get('response_len', 0)
    if resp_len > 400:
        analysis.append(f"{Colors.YELLOW}⚠ 回复过长 ({resp_len:.0f} tokens){Colors.END}")
    elif resp_len < 50:
        analysis.append(f"{Colors.RED}✗ 回复过短 ({resp_len:.0f} tokens){Colors.END}")
    else:
        analysis.append(f"{Colors.GREEN}✓ 回复长度适中 ({resp_len:.0f} tokens){Colors.END}")
    
    # KL散度分析
    kl = metrics.get('approx_kl', 0)
    if kl and kl > 0.5:
        analysis.append(f"{Colors.RED}✗ KL过高 ({kl:.3f}){Colors.END}")
    elif kl and kl > 0.1:
        analysis.append(f"{Colors.GREEN}✓ KL正常 ({kl:.3f}){Colors.END}")
    
    return analysis

def print_step_report(step, metrics, samples, analysis):
    """打印步骤报告"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{f'🚀 Step {step} | {timestamp}':^80}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    
    # 指标概览
    print(f"{Colors.CYAN}{Colors.BOLD}📊 关键指标:{Colors.END}")
    print(f"  平均奖励: {metrics.get('mean_reward', 0):.3f}")
    print(f"  解析率: {metrics.get('parse_rate', 0):.1%}")
    print(f"  格式率: {metrics.get('format_rate', 0):.1%}")
    print(f"  KL散度: {metrics.get('approx_kl', 0):.4f}")
    print(f"  回复长度: {metrics.get('response_len', 0):.1f} tokens")
    print(f"  GPU显存: {metrics.get('gpu_mem_gb', 0):.2f} GB")
    print()
    
    # 分析结果
    print(f"{Colors.CYAN}{Colors.BOLD}🔍 智能分析:{Colors.END}")
    for item in analysis:
        print(f"  {item}")
    print()
    
    # 样本展示
    if samples:
        print(f"{Colors.CYAN}{Colors.BOLD}💬 模型回复样例:{Colors.END}\n")
        for i, sample in enumerate(samples[:2]):
            question = sample.get('question', '')[:60] + "..." if len(sample.get('question', '')) > 60 else sample.get('question', '')
            response = sample.get('response', '')[:100] + "..." if len(sample.get('response', '')) > 100 else sample.get('response', '')
            pred = sample.get('pred', 'N/A')
            gold = sample.get('gold', 'N/A')
            reward = sample.get('reward', 0)
            
            status = Colors.GREEN if pred == gold else Colors.YELLOW
            
            print(f"  {Colors.BOLD}样本 {i+1}:{Colors.END}")
            print(f"    问题: {question}")
            print(f"    回复: {response[:80]}...")
            print(f"    预测: {pred} | 答案: {gold} | 奖励: {status}{reward:+.2f}{Colors.END}")
            print()
    
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

def monitor_and_analyze(steps=100):
    """监控训练并分析"""
    metrics_file = Path("outputs/ppo_math/metrics.jsonl")
    samples_file = Path("outputs/ppo_math/samples.jsonl")
    
    last_step = -1
    analysis_triggered = set()
    
    print(f"{Colors.GREEN}开始监控训练...{Colors.END}")
    print(f"监控文件: {metrics_file}")
    print(f"{Colors.YELLOW}按 Ctrl+C 停止监控{Colors.END}\n")
    
    try:
        while True:
            # 检查训练是否还在运行
            if not metrics_file.exists():
                time.sleep(1)
                continue
            
            # 读取最新指标
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    time.sleep(1)
                    continue
                
                # 获取最新step
                for line in reversed(lines):
                    try:
                        metrics = json.loads(line)
                        step = metrics.get('step', 0)
                        
                        # 每10个step分析一次
                        if step % 10 == 0 and step != last_step and step not in analysis_triggered:
                            # 读取对应样本
                            samples = []
                            if samples_file.exists():
                                with open(samples_file, 'r') as sf:
                                    for sline in sf:
                                        try:
                                            sample = json.loads(sline)
                                            if sample.get('step') == step:
                                                samples.append(sample)
                                        except:
                                            pass
                            
                            # 分析
                            step_data = {'metrics': metrics, 'samples': samples}
                            analysis = analyze_training_step(step_data)
                            
                            # 打印报告
                            print_step_report(step, metrics, samples, analysis)
                            
                            analysis_triggered.add(step)
                            last_step = step
                            
                            # 如果达到目标step，退出
                            if step >= steps:
                                print(f"{Colors.GREEN}训练完成！已达到目标 {steps} steps{Colors.END}")
                                return
                        
                        break
                    except Exception as e:
                        continue
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}监控已停止{Colors.END}")

def main():
    """主函数"""
    print_banner()
    
    # 解析参数
    steps = 100
    if '--steps' in sys.argv:
        idx = sys.argv.index('--steps')
        if idx + 1 < len(sys.argv):
            steps = int(sys.argv[idx + 1])
    
    # 清理旧数据
    print(f"{Colors.YELLOW}清理旧训练数据...{Colors.END}")
    os.makedirs("outputs/ppo_math", exist_ok=True)
    
    # 启动训练（后台）
    print(f"{Colors.GREEN}启动训练进程 (目标: {steps} steps)...{Colors.END}\n")
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    process = subprocess.Popen(
        [sys.executable, 'train_ppo.py', f'--ppo_steps={steps}'],
        stdout=open('training_output.log', 'w'),
        stderr=subprocess.STDOUT,
        env=env
    )
    
    print(f"{Colors.CYAN}训练PID: {process.pid}{Colors.END}")
    print(f"{Colors.CYAN}日志文件: training_output.log{Colors.END}\n")
    
    # 等待训练初始化
    print(f"{Colors.YELLOW}等待训练初始化 (10秒)...{Colors.END}\n")
    time.sleep(10)
    
    # 开始监控
    try:
        monitor_and_analyze(steps)
    except Exception as e:
        print(f"{Colors.RED}监控出错: {e}{Colors.END}")
    finally:
        # 确保训练进程结束
        if process.poll() is None:
            print(f"{Colors.YELLOW}正在停止训练进程...{Colors.END}")
            process.terminate()
            process.wait(timeout=10)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
演示：每隔10个step调用子agent分析训练情况
"""
import json
from pathlib import Path
import subprocess
import sys

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

def print_analysis_report(step, metrics, analysis_result):
    """打印分析报告"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{f'🔍 Step {step} - 子Agent智能分析报告':^80}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    
    # 原始指标
    print(f"{Colors.CYAN}{Colors.BOLD}📊 原始指标:{Colors.END}")
    print(f"  平均奖励: {metrics.get('mean_reward', 0):.3f}")
    print(f"  解析率: {metrics.get('parse_rate', 0):.1%}")
    print(f"  格式率: {metrics.get('format_rate', 0):.1%}")
    print(f"  KL散度: {metrics.get('approx_kl', 0):.4f}")
    print(f"  回复长度: {metrics.get('response_len', 0):.1f}")
    print(f"  训练时间: {metrics.get('step_time_sec', 0):.1f}s")
    print()
    
    # Agent分析结果
    print(f"{Colors.CYAN}{Colors.BOLD}🤖 子Agent分析结果:{Colors.END}\n")
    print(analysis_result)
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}\n")

def analyze_step_with_agent(step, metrics, samples):
    """调用子agent进行分析"""
    # 构建提示
    prompt = f"""
你是一位强化学习训练分析专家。请分析以下PPO训练数据并提供建议。

## 训练步骤 {step} 数据

### 关键指标
- 平均奖励: {metrics.get('mean_reward', 0):.3f}
- 解析成功率: {metrics.get('parse_rate', 0):.1%}
- 格式正确率: {metrics.get('format_rate', 0):.1%}
- KL散度: {metrics.get('approx_kl', 0):.4f}
- 回复平均长度: {metrics.get('response_len', 0):.1f} tokens
- 策略损失: {metrics.get('policy_loss', 0):.4f}
- 价值损失: {metrics.get('value_loss', 0):.4f}
- 训练耗时: {metrics.get('step_time_sec', 0):.1f}秒
- GPU显存: {metrics.get('gpu_mem_gb', 0):.2f} GB

### 模型回复样例（前2个）
"""
    
    for i, sample in enumerate(samples[:2]):
        response = sample.get('response', '')[:200]
        pred = sample.get('pred', 'N/A')
        gold = sample.get('gold', 'N/A')
        reward = sample.get('reward', 0)
        
        prompt += f"""
样例 {i+1}:
- 预测: {pred} | 答案: {gold} | 奖励: {reward:+.2f}
- 回复预览: {response}...
"""
    
    prompt += """

### 分析要求
请提供以下分析（每项2-3句话）：

1. **训练状态评估**: 当前训练是否正常？奖励趋势如何？
2. **格式遵循情况**: 模型是否学会使用 #### 格式？解析率说明了什么？
3. **奖励函数效果**: 奖励设置是否合理？有没有奖励黑客迹象？
4. **潜在问题**: 是否发现异常指标（如KL过高、损失过大等）？
5. **优化建议**: 针对当前状态，建议如何调整超参数或奖励函数？

请以简洁明了的方式输出分析结果。
"""
    
    # 这里模拟agent分析（实际应该调用真正的agent）
    # 由于是演示，我们生成基于规则的分析
    return generate_analysis(metrics, samples)

def generate_analysis(metrics, samples):
    """基于规则生成分析（模拟子agent）"""
    analysis = []
    
    # 1. 训练状态评估
    mean_reward = metrics.get('mean_reward', 0)
    if step == 0:
        analysis.append(f"{Colors.YELLOW}【训练状态】{Colors.END} 初始阶段，模型正在适应。奖励为负 ({mean_reward:.2f}) 是正常的，因为模型还未学会正确格式。")
    elif mean_reward < -0.3:
        analysis.append(f"{Colors.RED}【训练状态】{Colors.END} 奖励过低，模型可能没有学到有效策略。建议检查奖励函数或降低学习率。")
    elif mean_reward > 0.3:
        analysis.append(f"{Colors.GREEN}【训练状态】{Colors.END} 训练进展良好，模型开始获得正奖励。")
    else:
        analysis.append(f"{Colors.YELLOW}【训练状态】{Colors.END} 奖励接近0，模型在探索阶段。")
    
    # 2. 格式遵循
    parse_rate = metrics.get('parse_rate', 0)
    format_rate = metrics.get('format_rate', 0)
    
    if parse_rate < 0.3:
        analysis.append(f"{Colors.RED}【格式遵循】{Colors.END} 解析率仅 {parse_rate:.1%}，模型未学会 #### 格式。建议加强格式惩罚或在prompt中提供更多示例。")
    elif parse_rate < 0.7:
        analysis.append(f"{Colors.YELLOW}【格式遵循】{Colors.END} 解析率 {parse_rate:.1%}，部分学会格式但还不够稳定。继续训练应该会改善。")
    else:
        analysis.append(f"{Colors.GREEN}【格式遵循】{Colors.END} 解析率 {parse_rate:.1%}，模型已较好地学会 #### 格式。")
    
    if format_rate < parse_rate:
        analysis.append(f"{Colors.YELLOW}【格式细节】{Colors.END} 格式率 ({format_rate:.1%}) 低于解析率，说明有些回复虽然能被解析但不是标准格式。")
    
    # 3. 奖励函数
    correct_count = sum(1 for s in samples if s.get('pred') == s.get('gold'))
    total = len(samples)
    accuracy = correct_count / total if total > 0 else 0
    
    if accuracy < 0.2:
        analysis.append(f"{Colors.YELLOW}【奖励效果】{Colors.END} 样本准确率仅 {accuracy:.1%}，模型答案正确率较低。但当前重点是先学会格式，再追求正确率。")
    else:
        analysis.append(f"{Colors.GREEN}【奖励效果】{Colors.END} 样本准确率 {accuracy:.1%}，模型已开始学到一些解题能力。")
    
    # 4. 潜在问题
    kl = metrics.get('approx_kl', 0)
    if kl > 0.5:
        analysis.append(f"{Colors.RED}【异常警告】{Colors.END} KL散度过高 ({kl:.3f})！策略偏离参考模型太多，可能导致不稳定。建议增加KL惩罚或降低学习率。")
    
    resp_len = metrics.get('response_len', 0)
    if resp_len > 400:
        analysis.append(f"{Colors.YELLOW}【异常警告】{Colors.END} 回复过长 ({resp_len:.0f} tokens)，可能产生冗余内容。建议增加长度惩罚。")
    
    entropy = metrics.get('entropy', 0)
    if entropy > 8:
        analysis.append(f"{Colors.YELLOW}【异常警告】{Colors.END} 熵值过高 ({entropy:.2f})，模型输出过于随机。建议降低temperature。")
    
    # 5. 优化建议
    analysis.append(f"\n{Colors.CYAN}{Colors.BOLD}【优化建议】{Colors.END}")
    
    if parse_rate < 0.5:
        analysis.append(f"  1. 在System Prompt中增加更多 #### 格式的示例")
        analysis.append(f"  2. 考虑对非 #### 格式的回复给予更强的负奖励")
    
    if mean_reward < -0.5:
        analysis.append(f"  3. 检查奖励函数，确保答对时有足够的正奖励")
    
    if metrics.get('step_time_sec', 0) > 60:
        analysis.append(f"  4. 训练速度较慢，考虑减小 max_new_tokens 或增大 batch_size")
    
    analysis.append(f"  5. 继续观察后续steps，看指标是否改善")
    
    return "\n".join(analysis)

def main():
    """主函数"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'🤖 子Agent训练分析演示':^80}{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    print(f"{Colors.YELLOW}说明:{Colors.END}")
    print(f"  • 每隔10个step会自动调用子Agent分析训练情况")
    print(f"  • Agent会基于指标和样例生成诊断报告")
    print(f"  • 提供状态评估、问题诊断和优化建议")
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    # 读取step 0的数据进行演示
    metrics_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/metrics.jsonl")
    samples_file = Path("/home/uincy/projects/mywsl/ppo_math/outputs/ppo_math/samples.jsonl")
    
    if not metrics_file.exists():
        print(f"{Colors.RED}错误: 找不到训练数据{Colors.END}")
        return
    
    # 读取step 0的指标
    metrics = None
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('step') == 0:
                    metrics = data
                    break
            except:
                pass
    
    if not metrics:
        print(f"{Colors.RED}错误: 找不到step 0的数据{Colors.END}")
        return
    
    # 读取step 0的样本
    samples = []
    if samples_file.exists():
        with open(samples_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    if sample.get('step') == 0:
                        samples.append(sample)
                except:
                    pass
    
    # 调用Agent分析（演示）
    global step
    step = 0
    analysis = analyze_step_with_agent(0, metrics, samples)
    
    # 打印报告
    print_analysis_report(0, metrics, analysis)
    
    print(f"{Colors.GREEN}演示完成！{Colors.END}")
    print(f"\n{Colors.YELLOW}在实际训练中，这个分析会每隔10个step自动执行。{Colors.END}")
    print(f"{Colors.YELLOW}当前训练正在进行中，请等待后续steps完成...{Colors.END}\n")

if __name__ == "__main__":
    main()

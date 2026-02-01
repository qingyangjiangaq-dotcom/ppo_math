"""测试奖励函数，确保修复了奖励黑客问题。"""
import sys
sys.path.insert(0, '/home/uincy/projects/mywsl/ppo_math')

from reward import parse_pred, compute_reward


def test_parse_pred():
    """测试解析函数"""
    print("=" * 60)
    print("测试 parse_pred")
    print("=" * 60)
    
    # 测试1：严格格式 #### <数字>
    r1, ok1, fmt1 = parse_pred("所以答案是 #### 42")
    assert r1 == "42" and ok1 and fmt1, f"严格格式失败: {r1}, {ok1}, {fmt1}"
    print("✓ 严格格式 #### <数字> - 通过")
    
    # 测试2：答案标记
    r2, ok2, fmt2 = parse_pred("答案：100")
    assert r2 == "100" and ok2 and not fmt2, f"答案标记失败: {r2}, {ok2}, {fmt2}"
    print("✓ 答案标记 - 通过")
    
    # 测试3：极短回复的兜底逻辑
    r3, ok3, fmt3 = parse_pred("42")
    assert r3 == "42" and ok3 and not fmt3, f"极短回复失败: {r3}, {ok3}, {fmt3}"
    print("✓ 极短回复兜底 - 通过")
    
    # 测试4：长回复无严格格式 - 应该解析失败
    r4, ok4, fmt4 = parse_pred("Step 1: I have 5 apples. Step 2: I eat 10 apples.")
    assert not ok4, f"长回复无格式应该失败: {r4}, {ok4}, {fmt4}"
    print("✓ 长回复无严格格式 - 解析失败（正确）")
    
    # 测试5：完全无数字
    r5, ok5, fmt5 = parse_pred("我不知道答案")
    assert not ok5, f"无数字应该失败: {r5}, {ok5}, {fmt5}"
    print("✓ 无数字回复 - 解析失败（正确）")
    
    print("\n")


def test_compute_reward():
    """测试奖励函数 - 验证没有奖励黑客漏洞"""
    print("=" * 60)
    print("测试 compute_reward - 反奖励黑客检查")
    print("=" * 60)
    
    answer_text = "#### 42"
    
    # 测试1：答对且格式正确 - 应该获得 +1.5
    reward1, pred1, ok1, fmt1 = compute_reward("计算过程...\n#### 42", answer_text)
    assert reward1 == 1.5, f"答对+格式正确应为1.5, 得到{reward1}"
    print(f"✓ 答对+格式正确: reward={reward1} (期望1.5) - 通过")
    
    # 测试2：答对但格式错误 - 应该获得 +1.0
    reward2, pred2, ok2, fmt2 = compute_reward("答案是 42", answer_text)
    assert reward2 == 1.0, f"答对+格式错误应为1.0, 得到{reward2}"
    print(f"✓ 答对+格式错误: reward={reward2} (期望1.0) - 通过")
    
    # 测试3：答错 - 应该获得 -0.5
    reward3, pred3, ok3, fmt3 = compute_reward("#### 100", answer_text)
    assert reward3 == -0.5, f"答错应为-0.5, 得到{reward3}"
    print(f"✓ 答错: reward={reward3} (期望-0.5) - 通过")
    
    # 测试4：解析失败 - 应该获得 -1.0（重罚）
    reward4, pred4, ok4, fmt4 = compute_reward("乱七八糟的输出", answer_text)
    assert reward4 == -1.0, f"解析失败应为-1.0, 得到{reward4}"
    print(f"✓ 解析失败: reward={reward4} (期望-1.0) - 通过")
    
    # 测试5：【关键】验证没有长度奖励黑客漏洞
    # 以前：长回复+关键词+错误 = +0.1（正奖励！）
    # 现在：长回复+错误 = -0.5（负奖励）
    long_wrong_response = """
    因为这个问题很难，所以我需要仔细计算。
    首先，我要分析一下。然后，进行计算。
    步骤1: 设未知数。步骤2: 列方程。
    计算过程非常复杂，需要很多步骤。
    #### 999
    """
    reward5, pred5, ok5, fmt5 = compute_reward(long_wrong_response, answer_text)
    assert reward5 < 0, f"【关键】错误的长回复应该获得负奖励, 得到{reward5}"
    print(f"✓ 【关键】长+关键词+错误: reward={reward5} (期望<0, 旧版>0) - 通过")
    
    # 测试6：【关键】验证没有关键词奖励
    # 即使包含所有关键词，只要答错就是负分
    keyword_stuffing = "因为所以首先然后计算步骤step #### 999"
    reward6, pred6, ok6, fmt6 = compute_reward(keyword_stuffing, answer_text)
    assert reward6 == -0.5, f"【关键】关键词填充不应给分, 应为-0.5, 得到{reward6}"
    print(f"✓ 【关键】关键词填充+错误: reward={reward6} (期望-0.5) - 通过")
    
    # 测试7：【关键】验证即使长度达标，答错也是负分
    length_exploit = "x" * 200 + "\n#### 999"
    reward7, pred7, ok7, fmt7 = compute_reward(length_exploit, answer_text)
    assert reward7 == -0.5, f"【关键】长度达标但答错应为-0.5, 得到{reward7}"
    print(f"✓ 【关键】超长+错误: reward={reward7} (期望-0.5) - 通过")
    
    print("\n")


def test_reward_distribution():
    """测试奖励分布 - 确保模型有明确的学习信号"""
    print("=" * 60)
    print("测试奖励分布")
    print("=" * 60)
    
    answer_text = "#### 100"
    
    # 统计不同情况下的奖励
    rewards = {
        "答对+格式正确": compute_reward("#### 100", answer_text)[0],
        "答对+格式错误": compute_reward("答案：100", answer_text)[0],
        "答错": compute_reward("#### 99", answer_text)[0],
        "解析失败": compute_reward("???", answer_text)[0],
    }
    
    print("奖励分布:")
    for name, reward in rewards.items():
        print(f"  {name}: {reward:+.1f}")
    
    # 验证奖励层级：答对 > 答错 > 解析失败
    assert rewards["答对+格式正确"] > rewards["答对+格式错误"], "格式正确应有额外奖励"
    assert rewards["答对+格式错误"] > rewards["答错"], "答对应该比答错高"
    assert rewards["答错"] > rewards["解析失败"], "答错应该比解析失败好"
    
    print("✓ 奖励层级正确: 答对 > 答错 > 解析失败")
    print("\n")


if __name__ == "__main__":
    try:
        test_parse_pred()
        test_compute_reward()
        test_reward_distribution()
        
        print("=" * 60)
        print("✅ 所有测试通过！奖励函数已修复奖励黑客漏洞。")
        print("=" * 60)
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

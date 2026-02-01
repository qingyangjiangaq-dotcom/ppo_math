"""测试 System Prompt 修改是否正确"""
import sys
import re

# 直接从文件读取 SYSTEM_PROMPT，避免导入依赖
def get_system_prompt():
    """从 train_ppo.py 中提取 SYSTEM_PROMPT"""
    with open('/home/uincy/projects/mywsl/ppo_math/train_ppo.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 SYSTEM_PROMPT = ( ... ) 的模式
    pattern = r'SYSTEM_PROMPT = \((.*?)\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # 提取字符串内容，处理多行字符串拼接
        prompt_content = match.group(1)
        # 提取所有引号内的字符串
        strings = re.findall(r'["\'](.*?)["\']', prompt_content, re.DOTALL)
        result = ''.join(strings)
        if result:
            return result
    raise ValueError("无法提取 SYSTEM_PROMPT")

SYSTEM_PROMPT: str = get_system_prompt()


def test_system_prompt_structure():
    """测试 SYSTEM_PROMPT 包含必要的元素"""
    print("=" * 60)
    print("测试 SYSTEM_PROMPT 结构")
    print("=" * 60)
    
    # 检查包含关键元素
    assert "####" in SYSTEM_PROMPT, "必须包含 #### 标记"
    print("✓ 包含 #### 标记")
    
    assert "示例" in SYSTEM_PROMPT or "例子" in SYSTEM_PROMPT, "必须包含示例"
    print("✓ 包含示例/例子")
    
    assert "首先" in SYSTEM_PROMPT, "示例应包含分步推理词"
    print("✓ 示例包含分步推理词（首先）")
    
    assert "然后" in SYSTEM_PROMPT, "示例应包含连接词"
    print("✓ 示例包含连接词（然后）")
    
    # 检查示例格式正确
    assert "#### 4" in SYSTEM_PROMPT, "示例答案必须是 #### 4 格式"
    print("✓ 示例答案格式正确（#### 4）")
    
    # 检查指令清晰
    assert "格式" in SYSTEM_PROMPT, "应明确提到格式要求"
    print("✓ 明确提到格式要求")
    
    print("\n")


def test_system_prompt_length():
    """测试 SYSTEM_PROMPT 长度合理"""
    print("=" * 60)
    print("测试 SYSTEM_PROMPT 长度")
    print("=" * 60)
    
    length = len(SYSTEM_PROMPT)
    print(f"当前长度: {length} 字符")
    
    # Few-shot prompt 应该比普通 prompt 长
    assert length > 100, f"Few-shot prompt 应该比普通 prompt 长，当前只有 {length} 字符"
    print(f"✓ 长度合理 ({length} > 100)")
    
    # 但也不应过长（token 限制）
    assert length < 2000, f"Prompt 过长可能影响训练，当前 {length} 字符"
    print(f"✓ 未过长 ({length} < 2000)")
    
    print("\n")


def test_build_prompt_structure():
    """测试 build_prompt 函数结构（从代码中验证）"""
    print("=" * 60)
    print("测试 build_prompt 结构")
    print("=" * 60)
    
    # 读取 train_ppo.py 检查 build_prompt 实现
    with open('/home/uincy/projects/mywsl/ppo_math/train_ppo.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查 build_prompt 函数存在
    assert "def build_prompt" in content, "必须存在 build_prompt 函数"
    print("✓ build_prompt 函数存在")
    
    # 检查使用了 system role
    assert '"role": "system"' in content or "'role': 'system'" in content, "必须使用 system role"
    print("✓ 使用 system role")
    
    # 检查 SYSTEM_PROMPT 被使用
    assert "SYSTEM_PROMPT" in content, "必须使用 SYSTEM_PROMPT 变量"
    print("✓ 使用 SYSTEM_PROMPT 变量")
    
    # 检查 user role 正确
    assert '"role": "user"' in content or "'role': 'user'" in content, "必须使用 user role"
    print("✓ 使用 user role")
    
    # 检查使用了 apply_chat_template
    assert "apply_chat_template" in content, "必须使用 apply_chat_template"
    print("✓ 使用 apply_chat_template")
    
    print("\n")


def test_visual_anchor():
    """测试视觉锚点（#### 数字）是否突出"""
    print("=" * 60)
    print("测试视觉锚点")
    print("=" * 60)
    
    # #### 应该出现在换行后，作为最后一行
    lines = SYSTEM_PROMPT.split('\n')
    
    # 找到包含 #### 的行
    anchor_lines = [i for i, line in enumerate(lines) if '####' in line and '4' in line]
    assert len(anchor_lines) > 0, "必须包含 #### 4 示例"
    print("✓ 包含 #### 4 作为视觉锚点")
    
    # 检查 #### 4 是否在独立的一行（更突出）
    found_standalone = False
    for idx in anchor_lines:
        line = lines[idx].strip()
        if line.startswith('####'):
            print(f"✓ 示例答案在独立一行: '{line}'")
            found_standalone = True
            break
    
    if not found_standalone:
        print("⚠ 示例答案不在独立一行，但在内容中")
        # 非致命问题，只是说明
        pass
    
    print("\n")


def test_system_prompt_preview():
    """打印 SYSTEM_PROMPT 预览"""
    print("=" * 60)
    print("SYSTEM_PROMPT 预览")
    print("=" * 60)
    print(SYSTEM_PROMPT)
    print("\n")


if __name__ == "__main__":
    try:
        test_system_prompt_structure()
        test_system_prompt_length()
        test_build_prompt_structure()
        test_visual_anchor()
        test_system_prompt_preview()
        
        print("=" * 60)
        print("✅ 所有测试通过！SYSTEM_PROMPT 改进正确。")
        print("=" * 60)
        print("\n改进效果：")
        print("1. 从抽象描述 → 具体示例（Few-shot）")
        print("2. 增加视觉锚点（#### 4）供模型模仿")
        print("3. 明确格式要求（分步推理 + #### 数字）")
        print("4. 增强对 0.5B 小模型的约束力")
        
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

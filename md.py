import os

def code_to_md(project_path, output_file="output.md"):
    with open(output_file, 'w', encoding='utf-8') as md:  # 确保输出文件使用UTF-8
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    md.write(f"\n## {file}\n```python\n")
                    try:
                        # 尝试用UTF-8读取，若失败则用GBK带错误处理
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # 二次尝试用GBK读取
                        with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
                            content = f.read()
                    md.write(content)
                    md.write("\n```\n")

# 使用示例（请确保路径正确）
code_to_md('../毕设/project', 'project.md')  # 将'./project'替换为你的实际项目路径
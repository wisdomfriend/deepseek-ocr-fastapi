"""
模板工具函数
"""
from pathlib import Path
from typing import Optional


def get_template_path(template_name: str) -> Path:
    """获取模板文件路径"""
    template_dir = Path(__file__).parent.parent / "templates"
    return template_dir / template_name


def load_template(template_name: str) -> str:
    """加载模板文件内容"""
    template_path = get_template_path(template_name)
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


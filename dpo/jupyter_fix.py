# Jupyter环境下的alphageometry模块导入修复
# 直接在Jupyter notebook中运行此代码块

import os
import sys
from pathlib import Path

def fix_alphageometry_import():
    """修复Jupyter环境下的alphageometry模块导入问题"""
    
    print("🔧 修复Jupyter环境下的alphageometry模块导入")
    print("=" * 50)
    
    # 获取当前工作目录
    current_dir = Path.cwd()
    print(f"📁 当前工作目录: {current_dir}")
    
    # 尝试找到alphageometry模块
    possible_paths = [
        current_dir / "alphageometry",  # 当前目录下的alphageometry
        current_dir.parent / "alphageometry",  # 上级目录下的alphageometry
        current_dir / "alphageometry" / "alphageometry",  # 嵌套的alphageometry
    ]
    
    alphageometry_path = None
    for path in possible_paths:
        if path.exists():
            alphageometry_path = path
            print(f"✅ 找到alphageometry模块: {path}")
            break
    
    if alphageometry_path is None:
        print("❌ 未找到alphageometry模块")
        print("请确保您在正确的项目目录中运行此代码")
        print("可能的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    # 设置Python路径
    if str(alphageometry_path) not in sys.path:
        sys.path.insert(0, str(alphageometry_path))
        print(f"📝 已添加路径到sys.path: {alphageometry_path}")
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = str(alphageometry_path)
    print(f"🌍 已设置PYTHONPATH: {alphageometry_path}")
    
    # 测试导入
    try:
        import alphageometry
        print("✅ alphageometry模块导入成功!")
        return True
    except ImportError as e:
        print(f"❌ alphageometry模块导入失败: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 确保您在alphageometry项目根目录中运行")
        print("2. 检查alphageometry文件夹是否存在")
        print("3. 确保所有依赖已安装")
        return False

# 运行修复
success = fix_alphageometry_import()

if success:
    print("\n🎉 修复完成！现在可以正常使用alphageometry模块了")
    print("\n📋 使用示例:")
    print("```python")
    print("import alphageometry")
    print("# 现在可以正常导入和使用alphageometry模块了")
    print("```")
else:
    print("\n❌ 修复失败，请检查项目结构") 
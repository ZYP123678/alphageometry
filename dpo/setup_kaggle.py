#!/usr/bin/env python
# coding=utf-8

import os
import subprocess
import sys

def run_command(command):
    """运行命令并打印输出"""
    print(f"执行命令: {command}")
    os.system(command)

def setup_kaggle_environment():
    """设置 Kaggle 环境"""
    print("开始设置 Kaggle 环境...")
    
    # 清理 pip 缓存
    print("\n1. 清理 pip 缓存...")
    run_command("pip cache purge")
    
    # 升级基础工具
    print("\n2. 升级基础工具...")
    run_command("pip install --upgrade pip")
    run_command("pip install --upgrade packaging setuptools setuptools_scm wheel")
    
    # 安装基础包
    print("\n3. 安装基础包...")
    base_packages = [
        "numpy>=1.22.4",
        "scipy==1.10.0",
        "matplotlib==3.7.0",
        "tensorflow>=2.9.1",
        "jax>=0.3.13",
        "jaxlib>=0.3.13",
        "flax>=0.5.0",
        "optax>=0.1.2",
        "gin-config>=0.5.0",
        "absl-py>=1.0.0",
        "clu>=0.0.7",
        "sentencepiece>=0.1.96",
        "seqio>=0.0.7",
        "tensorflow-datasets>=4.5.2"
    ]
    
    for package in base_packages:
        print(f"安装 {package}...")
        run_command(f"pip install {package}")
    
    
    print("\n环境设置完成！")
    print("\n使用方法：")
    print("直接运行训练脚本：")
    print("python run_dpo.py")

if __name__ == "__main__":
    setup_kaggle_environment() 
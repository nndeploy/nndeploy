#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import sys

def copy_resources_to_android(resource_dir, android_assets_dir):
    """拷贝resources目录下的指定子目录到Android assets目录"""
    
    # 需要拷贝的子目录列表
    target_subdirs = ['audios', 'images', 'models', 'others', 'template', 'videos', 'workflow']
    
    if not os.path.exists(resource_dir):
        print(f"错误: 源目录 {resource_dir} 不存在")
        return False
    
    # 创建Android assets/resources目录（如果不存在）
    android_resources_dir = os.path.join(android_assets_dir, 'resources')
    os.makedirs(android_resources_dir, exist_ok=True)
    print(f"目标目录: {android_resources_dir}")
    
    success_count = 0
    
    for subdir in target_subdirs:
        source_path = os.path.join(resource_dir, subdir)
        target_path = os.path.join(android_resources_dir, subdir)
        
        if not os.path.exists(source_path):
            print(f"警告: 源目录 {source_path} 不存在，跳过")
            continue
        
        try:
            # 如果目标目录已存在，先删除
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
                print(f"已删除现有目录: {target_path}")
            
            # 拷贝目录
            shutil.copytree(source_path, target_path)
            print(f"已拷贝: {source_path} -> {target_path}")
            success_count += 1
            
        except Exception as e:
            print(f"错误: 拷贝 {source_path} 失败: {str(e)}")
    
    print(f"\n拷贝完成! 成功拷贝了 {success_count} 个目录")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="将resources目录下的指定子目录拷贝到Android应用的assets目录")
    parser.add_argument("--resource-dir", "-r", required=True,
                       help="源resources目录路径")
    parser.add_argument("--android-assets-dir", "-a", required=True,
                       help="Android assets目录路径")
    
    args = parser.parse_args()
    
    # 执行拷贝操作
    success = copy_resources_to_android(args.resource_dir, args.android_assets_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
    
# python3 tool/script/android_install_resouces.py -r  resources/ -a /home/always/github/public/nndeploy/app/android/app/src/main/assets


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob

def copy_dynamic_libraries(source_dir, target_dir):
    """
    将nndeploy构建目录下的动态库文件拷贝到Android应用的jniLibs目录中
    """
    # # 源目录和目标目录
    # source_dir = "nndeploy/build/nndeploy_2.6.1_Android_aarch64_Debug_Clang"
    # target_dir = "nndeploy/app/android/app/src/main/jniLibs/arm64-v8a"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 处理third_party目录
    third_party_dir = os.path.join(source_dir, "third_party")
    if os.path.exists(third_party_dir):
        print(f"处理third_party目录: {third_party_dir}")
        
        # 遍历third_party根目录下的文件夹
        for item in os.listdir(third_party_dir):
            item_path = os.path.join(third_party_dir, item)
            if os.path.isdir(item_path):
                # 查找该文件夹下的arm64-v8a目录
                arm64_dir = os.path.join(item_path, "arm64-v8a")
                if os.path.exists(arm64_dir):
                    print(f"  处理 {item}/arm64-v8a 目录")
                    # 拷贝arm64-v8a目录下的所有.so文件
                    so_files = glob.glob(os.path.join(arm64_dir, "*.so"))
                    for so_file in so_files:
                        filename = os.path.basename(so_file)
                        target_file = os.path.join(target_dir, filename)
                        print(f"    拷贝: {filename}")
                        shutil.copy2(so_file, target_file)
    
    # 处理其他目录（递归遍历所有.so文件）
    print(f"递归处理其他目录: {source_dir}")
    for root, dirs, files in os.walk(source_dir):
        # 跳过third_party目录，因为已经单独处理过了
        if "third_party" in root:
            continue
            
        for file in files:
            if file.endswith(".so"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # 获取相对路径用于显示
                rel_path = os.path.relpath(source_file, source_dir)
                print(f"  拷贝: {rel_path}")
                
                shutil.copy2(source_file, target_file)
    
    print(f"动态库拷贝完成，目标目录: {target_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("使用方法: python android_install_so.py <source_dir> <target_dir>")
        print("示例: python android_install_so.py nndeploy/build/nndeploy_2.6.1_Android_aarch64_Debug_Clang nndeploy/app/android/app/src/main/jniLibs/arm64-v8a")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    copy_dynamic_libraries(source_dir, target_dir)
    
# python3 tool/script/android_install_so.py /home/always/github/public/nndeploy/build/nndeploy_2.6.1_Android_aarch64_Debug_Clang /home/always/github/public/nndeploy/app/android/app/src/main/jniLibs/arm64-v8a
# python3 ../tool/script/android_install_so.py /home/always/github/public/nndeploy/build/nndeploy_2.6.1_Android_aarch64_Debug_Clang /home/always/github/public/nndeploy/app/android/app/src/main/jniLibs/arm64-v8a

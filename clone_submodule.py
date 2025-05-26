#!/usr/bin/env python3
import os
import configparser
import subprocess
import shutil

def parse_gitmodules(gitmodules_path='.gitmodules'):
    """解析.gitmodules文件,返回子模块信息列表"""
    config = configparser.ConfigParser()
    config.read(gitmodules_path)
    modules = []
    
    for section in config.sections():
        path = config[section]['path']
        url = config[section]['url']
        # 将https转换为ssh格式
        if url.startswith('https://github.com/'):
            url = 'git@github.com:' + url[19:]
            
        module_info = {
            'path': path,
            'url': url
        }
        
        # 检查是否有ref参数(用于指定特定的tag或branch)
        if 'ref' in config[section]:
            module_info['ref'] = config[section]['ref']
            
        # 检查是否有branch参数
        if 'branch' in config[section]:
            module_info['branch'] = config[section]['branch']
            
        modules.append(module_info)
        
    return modules

def clone_submodules(base_path='.'):
    """递归克隆所有子模块"""
    gitmodules_path = os.path.join(base_path, '.gitmodules')
    git_path = os.path.join(base_path, '.git')
    
    # 如果当前目录没有.gitmodules文件,直接返回
    if not os.path.exists(gitmodules_path):
        return
        
    modules = parse_gitmodules(gitmodules_path)
    
    for module in modules:
        path = module['path']
        url = module['url']
        full_path = os.path.join(base_path, path)
        
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
            
        # 克隆仓库
        clone_cmd = ['git', 'clone', url, full_path]
        subprocess.run(clone_cmd, check=True)

        # 获取子模块的正确commit
        if os.path.exists(git_path):
            git_modules_file = os.path.join(git_path, 'modules', path, 'HEAD')
            if os.path.exists(git_modules_file):
                with open(git_modules_file, 'r') as f:
                    content = f.read().strip()
                    # 处理引用格式
                    if content.startswith('ref: '):
                        # 提取分支名称 (例如 从 "ref: refs/heads/master" 提取 "master")
                        branch_name = content.split('/')[-1]
                        subprocess.run(['git', 'checkout', branch_name], cwd=full_path, check=True)
                    else:
                        # 如果是commit hash则直接checkout
                        subprocess.run(['git', 'checkout', content], cwd=full_path, check=True)
            else:
                # 如果指定了ref,切换到指定版本
                if 'ref' in module:
                    subprocess.run(['git', 'checkout', module['ref']], cwd=full_path, check=True)
                elif 'branch' in module:
                    subprocess.run(['git', 'checkout', module['branch']], cwd=full_path, check=True)

                # 获取子模块的commit hash
                git_config = os.path.join(git_path, 'config')
                if os.path.exists(git_config):
                    config = configparser.ConfigParser()
                    config.read(git_config)
                    submodule_section = f'submodule "{path}"'
                    if submodule_section in config.sections():
                        if 'commit' in config[submodule_section]:
                            commit_hash = config[submodule_section]['commit']
                            subprocess.run(['git', 'checkout', commit_hash], cwd=full_path, check=True)
            
        # 递归克隆子模块的子模块
        clone_submodules(full_path)

if __name__ == '__main__':
    # rm -rf third_party/*
    # subprocess.run(['rm', '-rf', 'third_party/*'], check=True)
    # # 通过git submodule update重新拉取所有子模块
    # subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)
    root_path = os.path.dirname(os.path.abspath(__file__))
    clone_submodules(root_path)
    # cd root_path
    os.chdir(root_path)
    # rm -rf third_party/*
    subprocess.run(['rm', '-rf', 'third_party/*'], check=True)
    # 通过git submodule update重新拉取所有子模块
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)

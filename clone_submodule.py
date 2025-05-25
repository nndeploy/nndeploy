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
        
        # 如果指定了ref,切换到指定版本
        if 'ref' in module:
            subprocess.run(['git', 'checkout', module['ref']], cwd=full_path, check=True)
        elif 'branch' in module:
            subprocess.run(['git', 'checkout', module['branch']], cwd=full_path, check=True)
            
        # 递归克隆子模块的子模块
        clone_submodules(full_path)

if __name__ == '__main__':
    clone_submodules()

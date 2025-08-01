# develop

## 

## 2025.08.01
```bash
# 确保使用Python 3.10
python3.10 --version  # 确认版本
# 或者使用pyenv管理Python版本
# pyenv install 3.10.12
# pyenv global 3.10.12

# 创建虚拟环境
python3.10 -m venv nndeploy_build_env
source nndeploy_build_env/bin/activate

# 升级基础工具
pip install --upgrade pip setuptools wheel
```
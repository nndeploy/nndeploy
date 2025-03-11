# ui

# 环境配置

## 创建python虚拟环境并安装flet
python -m venv venv
source venv/bin/activate
pip3 install flet

## 以应用形式运行ui
python3 mian.py --mode app

## 以网页形式运行ui
python3 mian.py --mode web

## 功能和结构

### 主界面

#### 基本介绍

+ nndeploy log

+ github

+ docs

+ about

+ 基本介绍

#### workflow

+ 当前唐广实现的即可

#### ui

+ 当前唐广实现的即可

class SDGraph：
  def __init__(self):
    pass

  def __call__(self):
    pass


sd_graph = SDGraph()

sd_graph.set_ouput_path("output.jpg")

mat = sd_graph("prompt")

ui = UI(sd_graph)

ui.show()

### Workflow - 对应comfyui

### app - 对应gradio



  
  

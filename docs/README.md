
## Install Sphinx
```
pip install sphinx
pip install sphinx_intl 
pip install sphinx_rtd_theme
pip install recommonmark
pip install sphinx-markdown-tables
```

## Build html
```
cd docs
# 中文文档
cd zh_cn
make html

sphinx-build -b gettext ./ build/gettext
sphinx-intl update -p ./build/gettext -l en
sphinx-build -b html -D language=en ./ build/html/en
# 英文文档
cd en
make html
```

## 文档提交注意事项
- 图片资源存放在docs/images中，需要用英文命名

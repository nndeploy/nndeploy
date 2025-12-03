# llm

运行nndeploy前端拉出来的llm工作流

## run_json (完整的运行整个工作流)

```bash
cd path/to/nndeploy

# Python CLI，保证当前工作目录下有resource资源
python3 demo/llm/demo.py --json_file resources/workflow/QwenMNN.json
# Python CLI，会将path/to/resources拷贝到当前工作目录
python3 demo.py --json_file path/to/resources/workflow/QwenMNN.json --resources path/to/resources

# C++ CLI，保证当前工作目录下有resource资源
./build/nndeploy_demo_llm --json_file resources/workflow/QwenMNN.json
# C++ CLI，会将path/to/resources拷贝到当前工作目录
.nndeploy_demo_llm --json_file path/to/resources/workflow/QwenMNN.json  --resources path/to/resources

# Result
A: 李白，唐代诗人，被誉为“诗仙”，以其高超的诗歌造诣和神奇的笔墨语言而受到世界的广泛喜爱。李白所作诗词广受欢迎，对于其深远影响和艺术成就的评价极高。李白生于唐代中叶，死于唐玄宗时期，活了600余岁，是唐代著名的诗人，被后人尊称为“诗圣”。
TimeProfiler: QwenMNN
-----------------------------------------------------------------------------------
name                     call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------
QwenMNN init()           1           950.283            950.283            0.000 
tokenizer_encode init()  1           86.864             86.864             0.000 
prefill_infer init()     1           863.319            863.319            0.000 
decode_infer init()      1           0.014              0.014              0.000 
tokenizer_decode init()  1           0.004              0.004              0.000 
QwenMNN run()            1           1153.749           1153.749           0.000 
Prefill_1 run()          1           29.306             29.306             0.000 
tokenizer_encode run()   1           0.154              0.154              0.000 
prefill_infer run()      1           26.082             26.082             0.000 
prefill_sampler run()    1           3.065              3.065              0.000 
Decode_2 run()           1           1124.438           1124.438           0.000 
decode_infer run()       82          902.054            11.001             0.000 
decode_sampler run()     82          221.435            2.700              0.000 
tokenizer_decode run()   82          0.313              0.004              0.000 
stream_out run()         82          0.358              0.004              0.000 
-----------------------------------------------------------------------------------
```

## run_json_remove_in_out_node（开发者自己有输入输出逻辑，移除工作流中的输入和输出节点）

```bash

cd path/to/nndeploy

# Python CLI，保证当前工作目录下有resource资源
python3 demo/llm/demo.py --json_file resources/workflow/QwenMNN.json --remove_in_out_node
# Python CLI，会将path/to/resources拷贝到当前工作目录
python3 demo/llm/demo.py --json_file path/to/resources/workflow/QwenMNN.json --remove_in_out_node  --resources path/to/resources

# C++ CLI，保证当前工作目录下有resource资源
./build/nndeploy_demo_llm --json_file resources/workflow/QwenMNN.json --remove_in_out_node
# C++ CLI，会将path/to/resources拷贝到当前工作目录
.nndeploy_demo_llm --json_file path/to/resources/workflow/QwenMNN.json --remove_in_out_node  --resources path/to/resources

# result
A: Jordan is a surname that was commonly used in the United States during the late 19th century. It was commonly used in the United States as a middle name, in order to not conflict with the common surname of Samuel.

Jordan was most likely given this surname by his father, who may have given Jordan this name simply because he had his own name and was not related to any other Jordan family. Today, Jordan is a popular last name in several countries around the world.

Jordan is also known as a childhood nickname for a person's first name.
TimeProfiler: QwenMNN
-----------------------------------------------------------------------------------
name                     call_times  sum cost_time(ms)  avg cost_time(ms)  gflops
-----------------------------------------------------------------------------------
QwenMNN init()           1           948.254            948.254            0.000 
tokenizer_encode init()  1           85.359             85.359             0.000 
prefill_infer init()     1           862.793            862.793            0.000 
decode_infer init()      1           0.012              0.012              0.000 
tokenizer_decode init()  1           0.004              0.004              0.000 
QwenMNN run()            1           1527.101           1527.101           0.000 
Prefill_1 run()          1           28.804             28.804             0.000 
tokenizer_encode run()   1           0.148              0.148              0.000 
prefill_infer run()      1           25.567             25.567             0.000 
prefill_sampler run()    1           3.084              3.084              0.000 
Decode_2 run()           1           1498.291           1498.291           0.000 
decode_infer run()       111         1205.737           10.862             0.000 
decode_sampler run()     111         291.288            2.624              0.000 
tokenizer_decode run()   111         0.446              0.004              0.000 
stream_out run()         111         0.470              0.004              0.000 
-----------------------------------------------------------------------------------
```
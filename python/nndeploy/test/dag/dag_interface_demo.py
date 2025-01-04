
class ClassificationResnetGraph(dag.Graph):
  def __init__(self, name, input, output):
    super().__init__(name, input, output)
    self.pre = self.createNode(preprocess.CvtColorResize) # 
    self.infer = self.createInfer(infer.Infer)
    self.post = self.createNode(classification.ClassificationPostProcess)

  # pre_param, infer_param, post_param 是 python/c++/json 的对象
  def forward(self, input, pre_param, infer_param, post_param):
    self.pre.setParam(pre_param)
    self.infer.setParam(infer_param)
    self.post.setParam(post_param)

    input = self.pre(input)
    input = self.infer(input)
    output = self.post(input)
    return output

def test_classification_resnet_graph():
  graph = ClassificationResnetGraph("classification")
  pre_param = preprocess.CvtColorResizeParam("pre.json")
  infer_param = infer.InferParam("infer.json")
  post_param = classification.ClassificationPostParam("post.json")
  input.set(cv2.imread("test.jpg"))
  output = graph.forward(input, pre_param, infer_param, post_param)
  print(output)

  # graph -> json
  graph.save("graph.json")
  # json -> graph
  graph = dag.Graph.load("graph.json")
  output = graph.forward(input, pre_param, infer_param, post_param)
  print(output)


class ClassificationResnetDemoGraph(dag.Graph):
  def __init__(self, name, input, output):
    super().__init__(name, input, output)
    self.decode = self.createNode(codec.Decode)
    self.deploy = self.createNode(ClassificationResnetGraph("classification_deploy"))
    self.encode = self.createNode(codec.Encode)

  def forward(self, input, decodec_param, pre_param, infer_param, post_param, encode_param):
    input = self.decode(input, decodec_param)
    output1 = self.deploy(input, pre_param, infer_param, post_param)
    output2 = self.encode(output1, encode_param)
    return output1, output2

def test_classification_resnet_demo_graph():
  graph = ClassificationResnetDemoGraph("classification")
  decodec_param = codec.DecodeParam("decode.json")
  pre_param = preprocess.CvtColorResizeParam("pre.json")
  infer_param = infer.InferParam("infer.json")
  post_param = classification.ClassificationPostParam("post.json")
  encode_param = codec.EncodeParam("encode.json")
  input.set("test.jpg")
  output1, output2 = graph.forward(input, decodec_param, pre_param, infer_param, post_param, encode_param)
  print(output1, output2)
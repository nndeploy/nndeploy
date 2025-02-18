
# from pydantic import Tag


class ClassificationResnetGraphV0(dag.Graph):
  def __init__(self, name, pre_param, infer_param, post_param):
    super().__init__(name)
    self.pre = self.createNode(preprocess.CvtColorResize, pre_param)
    self.infer = self.createInfer(infer.Infer, infer_param)
    self.post = self.createNode(classification.ClassificationPost, post_param)

  def forward(self, input):
    # preprocess
    # list[input] -> input
    # list[output] -> input - ["data"]
    input = self.pre(input, outpus_name = ["data"])
    # infer
    # list[input] -> input
    # list[output] -> input - [resnetv17_dense0_fwd]
    input = self.infer(input, outputs_name = ["resnetv17_dense0_fwd"])
    # postprocess
    # list[input] -> input
    # list[output] -> output
    output = self.post(input)
    return output

class ClassificationResnetGraphV1(dag.Graph):
  def __init__(self, name, pre_desc, infer_desc, post_desc):
    super().__init__(name)
    self.pre = self.createNode(preprocess.CvtColorResize, pre_desc)
    self.infer = self.createInfer(infer.Infer, infer_desc)
    self.post = self.createNode(classification.ClassificationPost, post_desc)

  def forward(self, input):
    input = self.pre(input)
    input = self.infer(input)
    output = self.post(input)
    return output

def test_classification_resnet_graph():
  pre_param = preprocess.CvtColorResizeParam("pre.json")
  infer_param = infer.InferParam("infer.json")
  post_param = classification.ClassificationPostParam("post.json")
  graph = ClassificationResnetGraphV0("classification", pre_param, infer_param, post_param)
  input = dag.Edge("input");
  input.set(cv2.imread("test.jpg"))
  output = graph.forward(input)
  print(output[0])

  # graph -> json
  graph.save("graph.json")
  # json -> graph
  graph = dag.Graph.load("graph.json")
  output = graph.forward(input)
  print(output)


class ClassificationResnetDemoGraph(dag.Graph, decodec_param, pre_desc, infer_desc, post_desc, encode_param):
  def __init__(self, name, input, output):
    super().__init__(name, input, output)
    self.decode = self.createNode(codec.Decode, decodec_param)
    self.deploy = self.createNode(ClassificationResnetGraphV0("classification_deploy", pre_desc, infer_desc, post_desc))
    self.encode = self.createNode(codec.Encode, decode_params)

  def forward(self, input):
    input = self.decode(input)
    output1 = self.deploy(input)
    output2 = self.encode(output1)
    return output1, output2

def test_classification_resnet_demo_graph():
  decodec_param = codec.DecodeParam("decode.json")
  pre_param = preprocess.CvtColorResizeParam("pre.json")
  infer_param = infer.InferParam("infer.json")
  post_param = classification.ClassificationPostParam("post.json")
  encode_param = codec.EncodeParam("encode.json")
  graph = ClassificationResnetDemoGraph("classification_demo", decodec_param, pre_param, infer_param, post_param, encode_param)
  input = dag.Edge("input")
  input.set("test.jpg")
  output1, output2 = graph.forward(input)
  print(output1)
  print(output2)

  # graph -> json
  graph.save("graph.json")
  # json -> graph
  graph = dag.Graph.load("graph.json")
  output1, output2 = graph.forward(input)
  print(output1)
  print(output2)

# TODO

# 组合以及扩增的实现方式
+ op::Node
  + ir::Node *
  + std::vector<ir::Initializer*>
  + op::ExecuteNode
  + inferShape
  + 一组对ExecuteNode接口的封装
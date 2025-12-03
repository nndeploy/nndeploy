# 子图
##	需求
  - 子图是一种节点容器, 可以容纳其他节点, 在流程图设计页面可通过拖拉的方式将其他节点拖入或移除其他节点.  
  - 当子图展开时, 外面的节点可连线子图的节点, 子图的节点也可连线外面的节点. 
  - 当子图收缩时, 里面的节点不可见, 但子图内部与外面的连线需要通过子图建立动态的端口与外面的节点连接, 当子图展开时, 动态的端口会被删除, 原来内部与外部的连线要恢复.
  - 当外面一个节点有多条线与子图中的多个节点连线时, 子图收缩后, 外面的这个节点只需一条线与这个子图相连
  - 删除收缩后的子图节点的动态端口与外面的连线, 再次展开节点, 此节点内之前与外面的相关连线要删除

  
##	实现方式

 - 子图的实现使用flowgram的子图功能,内置包含了节点的拖入与移除功能 .
 - 为实现子图收缩与展开连线依然能恢复, 需要保存连线的信息, 为避免引入第三方的状态库, 我们选择将连线的信息保存到子图的动态端口中, 具体实现方式如下: 
 - 当子图收缩时, 子图的出入线条信息保存到子图节点中, 每条线条包含源节点, 源端口, 目标节点,目的端口,为子图的展开操作恢复出入线条做好记录,  结构如下: 

  ```

  export interface IExpandInfo { //子图节点收缩后保存的连线信息

    inputLines: ILineEntity[], //输入连线
    outputLines: ILineEntity[] //输出连线
  }
  export interface ILineEntity {

    originFrom?: string, //origin node ID
    originFrom_name?: string, //origin node name : for debug
    originFromPort?: string | number,//origin node port

    oldFrom?: string; //old from node ID
    oldFrom_name?: string; //old from node name

    from: string,  //from node ID, when collapse subcavas node, outputline connnect to other node through subcavas
    from_name: string; //from node name

    oldFromPort?: string | number, //old from port ID
    fromPort: string | number, //from port ID, when collapse subcavas node, outputline connnect to other node through subcavas dynamic output port


    oldTo?: string;  //old to node ID
    oldTo_name?: string; //old to node name
    
    to: string,  //when collapse subcavas node, inputline reconnect to subcavas 
    to_name: string; //to node name

    oldToPort?: string | number, //old to port ID
    toPort: string | number, //to port ID, when collapse subcavas node, inputline reconnect to subcavas input dynamic port


    originTo?: string,
    originTo_name?: string;
    originToPort?: string | number,

    type_: string,
    desc_: string,

    // [key:string]:any
  }
  ```

   - 收缩操作逻辑请参考nndeploy\app\workflow\src\pages\components\flow\form-header\index.tsx中的 function shrimpNode() 
   - 展开操作逻辑请参考nndeploy\app\workflow\src\pages\components\flow\form-header\index.tsx中的 function expandNode() 

   - 当删除连线时 通过判断连线的from, 与to两个节点是否为子图节点, 若为子图节点, 将子图保存的这条线相关的IExpandInfo的连线信息一并删除, 这样子图再次展开后, 子图在收缩时保存的内部节点与外部节点的连线就不用重建了. 相关代码请参考nndeploy\app\workflow\src\hooks\use-editor-props.tsx中的canDeleteLine()函数

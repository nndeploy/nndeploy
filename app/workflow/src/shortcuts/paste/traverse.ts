// traverse value type - 遍历值类型
export type TraverseValue = any;

// traverse node interface - 遍历节点接口
export interface TraverseNode {
  value: TraverseValue; // node value - 节点值
  container?: TraverseValue; // parent container - 父容器
  parent?: TraverseNode; // parent node - 父节点
  key?: string; // object key - 对象键名
  index?: number; // array index - 数组索引
}

// traverse context interface - 遍历上下文接口
export interface TraverseContext {
  node: TraverseNode; // current node - 当前节点
  setValue: (value: TraverseValue) => void; // set value function - 设置值函数
  getParents: () => TraverseNode[]; // get parents function - 获取父节点函数
  getPath: () => Array<string | number>; // get path function - 获取路径函数
  getStringifyPath: () => string; // get string path function - 获取字符串路径函数
  deleteSelf: () => void; // delete self function - 删除自身函数
}

// traverse handler type - 遍历处理器类型
export type TraverseHandler = (context: TraverseContext) => void;

/**
 * traverse object deeply and handle each value - 深度遍历对象并处理每个值
 * @param value traverse target - 遍历目标
 * @param handle handler function - 处理函数
 */
export const traverse = <T extends TraverseValue = TraverseValue>(
  value: T,
  handler: TraverseHandler | TraverseHandler[]
): T => {
  const traverseHandler: TraverseHandler = Array.isArray(handler)
    ? (context: TraverseContext) => {
        handler.forEach((handlerFn) => handlerFn(context));
      }
    : handler;
  TraverseUtils.traverseNodes({ value }, traverseHandler);
  return value;
};

namespace TraverseUtils {
  /**
   * traverse nodes deeply and handle each value - 深度遍历节点并处理每个值
   * @param node traverse node - 遍历节点
   * @param handle handler function - 处理函数
   */
  export const traverseNodes = (node: TraverseNode, handle: TraverseHandler): void => {
    const { value } = node;
    if (!value) {
      // handle null value - 处理空值
      return;
    }
    if (Object.prototype.toString.call(value) === '[object Object]') {
      // traverse object properties - 遍历对象属性
      Object.entries(value).forEach(([key, item]) =>
        traverseNodes(
          {
            value: item,
            container: value,
            key,
            parent: node,
          },
          handle
        )
      );
    } else if (Array.isArray(value)) {
      // traverse array elements from end to start - 从末尾开始遍历数组元素
      for (let index = value.length - 1; index >= 0; index--) {
        const item: string = value[index];
        traverseNodes(
          {
            value: item,
            container: value,
            index,
            parent: node,
          },
          handle
        );
      }
    }
    const context: TraverseContext = createContext({ node });
    handle(context);
  };

  /**
   * create traverse context - 创建遍历上下文
   * @param node traverse node - 遍历节点
   */
  const createContext = ({ node }: { node: TraverseNode }): TraverseContext => ({
    node,
    setValue: (value: unknown) => setValue(node, value),
    getParents: () => getParents(node),
    getPath: () => getPath(node),
    getStringifyPath: () => getStringifyPath(node),
    deleteSelf: () => deleteSelf(node),
  });

  /**
   * set node value - 设置节点值
   * @param node traverse node - 遍历节点
   * @param value new value - 新值
   */
  const setValue = (node: TraverseNode, value: unknown) => {
    // handle empty value - 处理空值
    if (!value || !node) {
      return;
    }
    node.value = value;
    // get container info from parent scope - 从父作用域获取容器信息
    const { container, key, index } = node;
    if (key && container) {
      container[key] = value;
    } else if (typeof index === 'number') {
      container[index] = value;
    }
  };

  /**
   * get parent nodes - 获取父节点列表
   * @param node traverse node - 遍历节点
   */
  const getParents = (node: TraverseNode): TraverseNode[] => {
    const parents: TraverseNode[] = [];
    let currentNode: TraverseNode | undefined = node;
    while (currentNode) {
      parents.unshift(currentNode);
      currentNode = currentNode.parent;
    }
    return parents;
  };

  /**
   * get node path - 获取节点路径
   * @param node traverse node - 遍历节点
   */
  const getPath = (node: TraverseNode): Array<string | number> => {
    const path: Array<string | number> = [];
    const parents = getParents(node);
    parents.forEach((parent) => {
      if (parent.key) {
        path.unshift(parent.key);
      } else if (parent.index) {
        path.unshift(parent.index);
      }
    });
    return path;
  };

  /**
   * get stringify path - 获取字符串路径
   * @param node traverse node - 遍历节点
   */
  const getStringifyPath = (node: TraverseNode): string => {
    const path = getPath(node);
    return path.reduce((stringifyPath: string, pathItem: string | number) => {
      if (typeof pathItem === 'string') {
        const re = /\W/g;
        if (re.test(pathItem)) {
          // handle special characters - 处理特殊字符
          return `${stringifyPath}["${pathItem}"]`;
        }
        return `${stringifyPath}.${pathItem}`;
      } else {
        return `${stringifyPath}[${pathItem}]`;
      }
    }, '');
  };

  /**
   * delete current node - 删除当前节点
   * @param node traverse node - 遍历节点
   */
  const deleteSelf = (node: TraverseNode): void => {
    const { container, key, index } = node;
    if (key && container) {
      delete container[key];
    } else if (typeof index === 'number') {
      container.splice(index, 1);
    }
  };
}

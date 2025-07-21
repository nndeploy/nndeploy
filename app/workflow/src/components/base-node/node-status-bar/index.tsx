import { useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context"
import { NodeStatusHeader } from "./header";
import styles from './index.module.scss';
import { WorkflowNodeStatus } from "./type";
import { IconSpin } from "@douyinfe/semi-icons";
import classnames from 'classnames';
import { IconSuccessFill } from "../../../assets/icon-success";
import { IconWarningFill } from "../../../assets/icon-warning";
import { Tag } from "@douyinfe/semi-ui";

export const NodeStatusBar: React.FC<any> = (props) => {
  const flowEnviroment = useFlowEnviromentContext()

  const { flowNodesRunningStatus } = flowEnviroment

  const nodeRender = useNodeRender();
  /**
   * It can only be used when nodeEngine is enabled
   * 只有在节点引擎开启时候才能使用表单
   */
  const form = nodeRender.form;

  const name = form?.getValueIn('name_')

  const nodeInfo = flowNodesRunningStatus[name]

  if (!nodeInfo) {
    return <></>
  }

  const isNodeIdle = nodeInfo.status === WorkflowNodeStatus.IDLE;
  const isNodePending = nodeInfo.status === WorkflowNodeStatus.PENDING;
  const isNodeRunning = nodeInfo.status === WorkflowNodeStatus.RUNNING;
  const isNodeDone = nodeInfo.status === WorkflowNodeStatus.DONE;

  const tagColor = () => {
    if (isNodeDone) {
      return styles.nodeStatusSucceed;
    }
    if (isNodePending) {
      return styles.nodeStatusProcessing;
    }
    return ""
  }

  const renderIcon = () => {
    if (isNodePending) {
      return <IconSpin spin className={classnames(styles.icon, styles.processing)} />;
    }
    if (isNodeDone) {
      return <IconSuccessFill />;
    }
    return <IconWarningFill className={classnames(tagColor(), styles.round)} />;
  };

  const renderCost = () => (
    <Tag size="small" className={tagColor()}>
      {nodeInfo.time.toFixed(2)}
    </Tag>
  );

  return <NodeStatusHeader
    header={
      <>
        {renderIcon()}
        {nodeInfo.status}
        {renderCost()}
      </>
    }
  >
    <div className={styles.container}></div>
  </NodeStatusHeader>


}
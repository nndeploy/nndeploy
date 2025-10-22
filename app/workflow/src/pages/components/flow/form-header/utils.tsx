import { type FlowNodeEntity } from '@flowgram.ai/free-layout-editor';


import { Icon } from './styles';
import { FlowNodeRegistry } from '../../../../typings';

export const getIcon = (node: FlowNodeEntity) => {
  const icon = node.getNodeRegistry<FlowNodeRegistry>().info?.icon;
  if (!icon) return null;
  return <Icon src={icon} />;
};

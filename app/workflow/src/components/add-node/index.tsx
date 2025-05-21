import { Button } from '@douyinfe/semi-ui';
import { IconPlus } from '@douyinfe/semi-icons';

import { useAddNode } from './use-add-node';

export const AddNode = (props: { disabled: boolean }) => {
  const addNode = useAddNode();
  return (
    <Button
      icon={<IconPlus />}
      color="highlight"
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
      disabled={props.disabled}
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        addNode(rect);
      }}
    >
      Add Node
    </Button>
  );
};

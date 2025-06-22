import { Field, FieldRenderProps } from '@flowgram.ai/free-layout-editor';
import { Typography, Button } from '@douyinfe/semi-ui';
import { IconSmallTriangleDown, IconSmallTriangleLeft } from '@douyinfe/semi-icons';
import { getIcon } from './utils';
import { Header, Operators, Title , SecondTitle} from './styles';
import { useIsSidebar, useNodeRenderContext } from '../../../../hooks';
import { NodeMenu } from '../../../../components/node-menu';
import { Feedback } from '../../../../form-components';

const { Text } = Typography;

export function FormHeader() {
  const { node, expanded, toggleExpand, readonly, deleteNode } = useNodeRenderContext();
  const isSidebar = useIsSidebar();
  const handleExpand = (e: React.MouseEvent) => {
    toggleExpand();
    e.stopPropagation(); // Disable clicking prevents the sidebar from opening
  };

  return (
    <>
    <Header>
      {/* {getIcon(node)} */}
      <Title>
        {/* <Field name="key_">
          {({ field: { value, onChange }, fieldState }: FieldRenderProps<string>) => (
            <div style={{ height: 24 }}>
              <Text ellipsis={{ showTooltip: true }} >{value}</Text>
              <Feedback errors={fieldState?.errors} />
            </div>
          )}
        </Field> */}
         <Field name="name_">
          {({ field: { value, onChange }, fieldState }: FieldRenderProps<string>) => (
            <div style={{ height: 24 }}>
              <Text ellipsis={{ showTooltip: true }}>{value}</Text>
              <Feedback errors={fieldState?.errors} />
            </div>
          )}
        </Field>
      </Title>
      {node.renderData.expandable && !isSidebar && (
        <Button
          type="primary"
          icon={expanded ? <IconSmallTriangleDown /> : <IconSmallTriangleLeft />}
          size="small"
          theme="borderless"
          onClick={handleExpand}
        />
      )}
      {readonly ? undefined : (
        <Operators>
          <NodeMenu node={node} deleteNode={deleteNode} />
        </Operators>
      )}
    </Header>
    {/* <SecondTitle> 
     
    </SecondTitle> */}
    </>
  );
}

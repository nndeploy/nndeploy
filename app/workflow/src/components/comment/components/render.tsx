import { FC, useState } from 'react';

import {
  Field,
  FieldRenderProps,
  FlowNodeFormData,
  Form,
  FormModelV2,
  useNodeRender,
  WorkflowNodeEntity,
} from '@flowgram.ai/free-layout-editor';

import { useOverflow } from '../hooks/use-overflow';
import { useModel } from '../hooks/use-model';
import { useSize } from '../hooks';
import { CommentEditorFormField } from '../constant';
import { MoreButton } from './more-button';
import { CommentEditor } from './editor';
import { ContentDragArea } from './content-drag-area';
import { CommentContainer } from './container';
import { BorderArea } from './border-area';
import { Switch } from '@douyinfe/semi-ui';
import CodeBlock from '../../../pages/components/flow/nodeRegistry/CodeBlock';

export const CommentRender: FC<{
  node: WorkflowNodeEntity;
}> = (props) => {
  const { node } = props;
  const model = useModel();

  const { selected: focused, selectNode, nodeRef, deleteNode } = useNodeRender();

  const formModel = node.getData(FlowNodeFormData).getFormModel<FormModelV2>();
  const formControl = formModel?.formControl;

  const { width, height, onResize } = useSize({ width: 200, height: 180 });
  const { overflow, updateOverflow } = useOverflow({ model, height });

  const [language, setLanguage] = useState('text');

  return (
    <div
      className="workflow-comment"
      style={{
        width,
        height,
      }}
      ref={nodeRef}
      data-node-selected={String(focused)}
      onMouseEnter={updateOverflow}
      onMouseDown={(e) => {
        setTimeout(() => {
          // 防止 selectNode 拦截事件，导致 slate 编辑器无法聚焦
          selectNode(e);
          // eslint-disable-next-line @typescript-eslint/no-magic-numbers -- delay
        }, 20);
      }}
    >

      <Form control={formControl}>
        <>
          {/* 背景 */}
          <CommentContainer focused={focused} style={{ height }}>


            <div className='comment-content-area'>
              <Field name={CommentEditorFormField.Note} >
                {({ field }: FieldRenderProps<string>) => {


                  return language == 'text' ? <>
                    {/** 编辑器 */}
                    <CommentEditor model={model} value={field.value} onChange={field.onChange} />

                  </>
                    :
                    <CodeBlock
                      code={field.value}
                      language={language}
                    />
                }}
              </Field>
              {/* 内容拖拽区域（点击后隐藏） */}
              <ContentDragArea model={model} focused={focused} overflow={overflow} />
              {/* 更多按钮 */}
              <MoreButton node={node} focused={focused} deleteNode={deleteNode} />
            </div>

            <div className={'comment-operte-area'}>
              <Switch
                defaultChecked={language == 'markdown'}

                // checked={language == 'text'}
                onChange={(v) => {
                  setLanguage(v ? 'markdown' : 'text')
                }}
              />
            </div>


          </CommentContainer>
          {/* 边框 */}
          <BorderArea model={model} overflow={overflow} onResize={onResize} />
        </>
      </Form>
    </div>
  );
};

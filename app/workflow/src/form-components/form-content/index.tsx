import React from 'react';

import { FlowNodeRegistry } from '@flowgram.ai/free-layout-editor';

import { useIsSidebar, useNodeRenderContext } from '../../hooks';
import { FormTitleDescription, FormWrapper } from './styles';

/**
 * @param props
 * @constructor
 */
export function FormContent(props: { children?: React.ReactNode }) {
  const { node, expanded } = useNodeRenderContext();
  const isSidebar = useIsSidebar();
  const registry = node.getNodeRegistry<FlowNodeRegistry>();
  return (
    <FormWrapper>
      {expanded ? (
        <>
          {isSidebar && <FormTitleDescription>{registry.info?.description}</FormTitleDescription>}
          {props.children}
        </>
      ) : undefined}
    </FormWrapper>
  );
}

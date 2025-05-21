import { useState } from 'react';

import { NodeRenderReturnType } from '@flowgram.ai/free-layout-editor';

import { SidebarContext } from '../../context';

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [nodeRender, setNodeRender] = useState<NodeRenderReturnType | undefined>();
  return (
    <SidebarContext.Provider value={{ visible: !!nodeRender, nodeRender, setNodeRender }}>
      {children}
    </SidebarContext.Provider>
  );
}

import { useState, useEffect, forwardRef, useRef, useImperativeHandle } from 'react';

import { useRefresh } from '@flowgram.ai/free-layout-editor';
import { useClientContext } from '@flowgram.ai/free-layout-editor';
import { Tooltip, IconButton, Divider, Button } from '@douyinfe/semi-ui';
import { IconUndo, IconRedo, IconSetting } from '@douyinfe/semi-icons';

import { AddNode } from '../add-node';
import { ZoomSelect } from './zoom-select';
import { SwitchLine } from './switch-line';
import { ToolContainer, ToolSection } from './styles';
import { Save } from './save';
import { Run } from './run';
import { Readonly } from './readonly';
import { MinimapSwitch } from './minimap-switch';
import { Minimap } from './minimap';
import { Interactive } from './interactive';
import { FitView } from './fit-view';
import { Comment } from './comment';
import { AutoLayout } from './auto-layout';
import { IconConfig } from '@douyinfe/semi-icons-lab';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { Config } from './config';
import Log from './log';

export interface AutoLayoutHandle {
  autoLayout: () => void;
}

export const DemoTools = forwardRef<AutoLayoutHandle>((props, ref) => {
  const { history, playground } = useClientContext();
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [minimapVisible, setMinimapVisible] = useState(true);
  useEffect(() => {
    const disposable = history.undoRedoService.onChange(() => {
      setCanUndo(history.canUndo());
      setCanRedo(history.canRedo());
    });
    return () => disposable.dispose();
  }, [history]);
  const refresh = useRefresh();

  useEffect(() => {
    const disposable = playground.config.onReadonlyOrDisabledChange(() => refresh());
    return () => disposable.dispose();
  }, [playground]);


  const autoLayoutRef = useRef<any>(null);

  useImperativeHandle(ref, () => ({
    autoLayout: () => {
      autoLayoutRef.current?.props?.onClick();
    }
  }));


  return (
    <ToolContainer className="demo-free-layout-tools">
      <ToolSection>
        <Interactive />
        <AutoLayout ref={autoLayoutRef} />
        <SwitchLine />
        <ZoomSelect />
        <FitView />
        <MinimapSwitch minimapVisible={minimapVisible} setMinimapVisible={setMinimapVisible} />
        <Minimap visible={minimapVisible} />
        <Readonly />
        <Comment />
        <Tooltip content="Undo">
          <IconButton
            type="tertiary"
            theme="borderless"
            icon={<IconUndo />}
            disabled={!canUndo || playground.config.readonly}
            onClick={() => history.undo()}
          />
        </Tooltip>
        <Tooltip content="Redo">
          <IconButton
            type="tertiary"
            theme="borderless"
            icon={<IconRedo />}
            disabled={!canRedo || playground.config.readonly}
            onClick={() => history.redo()}
          />
        </Tooltip>
        <Log />
        <Divider layout="vertical" style={{ height: '16px' }} margin={3} />
        {/* <AddNode disabled={playground.config.readonly} />
        <Divider layout="vertical" style={{ height: '16px' }} margin={3} /> */}

        {/* <Tooltip content="config">
          <IconButton
            type="tertiary"
            theme="borderless"
            icon={<IconSetting />}
           
            onClick={() => history.undo()}
          />
        </Tooltip> */}
        <Config />
        <Save disabled={playground.config.readonly} />
        <Run />
      </ToolSection>
    </ToolContainer>
  );
});

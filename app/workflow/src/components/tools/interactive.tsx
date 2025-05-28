import { useEffect, useState } from 'react';

import {
  usePlaygroundTools,
  type InteractiveType as IdeInteractiveType,
} from '@flowgram.ai/free-layout-editor';
import { Tooltip, Popover } from '@douyinfe/semi-ui';

import { MousePadSelector } from './mouse-pad-selector';

export const CACHE_KEY = 'workflow_prefer_interactive_type';
export const SHOW_KEY = 'show_workflow_interactive_type_guide';
export const IS_MAC_OS = /(Macintosh|MacIntel|MacPPC|Mac68K|iPad)/.test(navigator.userAgent);

export const getPreferInteractiveType = () => {
  const data = localStorage.getItem(CACHE_KEY) as string;
  if (data && [InteractiveType.Mouse, InteractiveType.Pad].includes(data as InteractiveType)) {
    return data;
  }
  return IS_MAC_OS ? InteractiveType.Pad : InteractiveType.Mouse;
};

export const setPreferInteractiveType = (type: InteractiveType) => {
  localStorage.setItem(CACHE_KEY, type);
};

export enum InteractiveType {
  Mouse = 'MOUSE',
  Pad = 'PAD',
}

export const Interactive = () => {
  const tools = usePlaygroundTools();
  const [visible, setVisible] = useState(false);

  const [interactiveType, setInteractiveType] = useState<InteractiveType>(
    () => getPreferInteractiveType() as InteractiveType
  );

  const [showInteractivePanel, setShowInteractivePanel] = useState(false);

  const mousePadTooltip =
    interactiveType === InteractiveType.Mouse ? 'Mouse-Friendly' : 'Touchpad-Friendly';

  useEffect(() => {
    tools.setMouseScrollDelta((zoom) => zoom / 20);

    // read from localStorage
    const preferInteractiveType = getPreferInteractiveType();
    tools.setInteractiveType(preferInteractiveType as IdeInteractiveType);
  }, []);

  const handleClose = () => {
    setVisible(false);
  };

  return (
    <Popover trigger="custom" position="top" visible={visible} onClickOutSide={handleClose}>
      <Tooltip
        content={mousePadTooltip}
        style={{ display: showInteractivePanel ? 'none' : 'block' }}
      >
        <div className="workflow-toolbar-interactive">
          <MousePadSelector
            value={interactiveType}
            onChange={(value) => {
              setInteractiveType(value);
              setPreferInteractiveType(value);
              tools.setInteractiveType(value as unknown as IdeInteractiveType);
            }}
            onPopupVisibleChange={setShowInteractivePanel}
            containerStyle={{
              border: 'none',
              height: '32px',
              width: '32px',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '2px',
              padding: '4px',
              borderRadius: 'var(--small, 6px)',
            }}
            iconStyle={{
              margin: '0',
              width: '16px',
              height: '16px',
            }}
            arrowStyle={{
              width: '12px',
              height: '12px',
            }}
          />
        </div>
      </Tooltip>
    </Popover>
  );
};

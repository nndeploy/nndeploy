import { useState } from 'react';

import { usePlayground, usePlaygroundTools } from '@flowgram.ai/free-layout-editor';
import { Divider, Dropdown } from '@douyinfe/semi-ui';

import { SelectZoom } from './styles';

export const ZoomSelect = () => {
  const tools = usePlaygroundTools();
  const playground = usePlayground();
  const [dropDownVisible, openDropDown] = useState(false);
  return (
    <Dropdown
      position="top"
      trigger="custom"
      visible={dropDownVisible}
      onClickOutSide={() => openDropDown(false)}
      render={
        <Dropdown.Menu>
          <Dropdown.Item onClick={() => tools.zoomin()}>Zoom in</Dropdown.Item>
          <Dropdown.Item onClick={() => tools.zoomout()}>Zoom out</Dropdown.Item>
          <Divider layout="horizontal" />
          <Dropdown.Item onClick={() => playground.config.updateZoom(0.5)}>
            Zoom to 50%
          </Dropdown.Item>
          <Dropdown.Item onClick={() => playground.config.updateZoom(1)}>
            Zoom to 100%
          </Dropdown.Item>
          <Dropdown.Item onClick={() => playground.config.updateZoom(1.5)}>
            Zoom to 150%
          </Dropdown.Item>
          <Dropdown.Item onClick={() => playground.config.updateZoom(2.0)}>
            Zoom to 200%
          </Dropdown.Item>
        </Dropdown.Menu>
      }
    >
      <SelectZoom onClick={() => openDropDown(true)}>{Math.floor(tools.zoom * 100)}%</SelectZoom>
    </Dropdown>
  );
};

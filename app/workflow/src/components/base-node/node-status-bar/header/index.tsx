/**
 * Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 * SPDX-License-Identifier: MIT
 */

import React, { useState } from 'react';

import classNames from 'classnames';
import { IconChevronDown } from '@douyinfe/semi-icons';

import { useNodeRenderContext } from '../../../../hooks';

import styles from './index.module.scss';

interface NodeStatusBarProps {
  header?: React.ReactNode;
  defaultShowDetail?: boolean;
  extraBtns?: React.ReactNode[];
}

export const NodeStatusHeader: React.FC<React.PropsWithChildren<NodeStatusBarProps>> = ({
  header,
  defaultShowDetail,
  children,
  extraBtns = [],
}) => {
  const [showDetail, setShowDetail] = useState(defaultShowDetail);
  const { selectNode } = useNodeRenderContext();

  const handleToggleShowDetail = (e: React.MouseEvent) => {
    e.stopPropagation();
    selectNode(e);
    setShowDetail(!showDetail);
  };

  return (
    <div
      className={styles['node-status-header']}
      // 必须要禁止 down 冒泡，防止判定圈选和 node hover（不支持多边形）
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div
        className={classNames(
          styles['node-status-header-content'],
          showDetail && styles['node-status-header-content-opened']
        )}
        // 必须要禁止 down 冒泡，防止判定圈选和 node hover（不支持多边形）
        onMouseDown={(e) => e.stopPropagation()}
        // 其他事件统一走点击事件，且也需要阻止冒泡
        onClick={handleToggleShowDetail}
      >
        <div className={styles['status-title']}>
          {header}
          {extraBtns.length > 0 ? extraBtns : null}
        </div>
        <div className={styles['status-btns']}>
          <IconChevronDown
            className={classNames({
              [styles['is-show-detail']]: showDetail,
            })}
          />
        </div>
      </div>
      {showDetail ? children : null}
    </div>
  );
};

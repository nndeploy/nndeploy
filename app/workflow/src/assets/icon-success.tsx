/**
 * Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 * SPDX-License-Identifier: MIT
 */

interface Props {
  className?: string;
  style?: React.CSSProperties;
}

export const IconSuccessFill = ({ className, style }: Props) => (
  <svg
    className={className}
    style={style}
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    fill="none"
    viewBox="0 0 20 20"
  >
    <g clipPath="url(#icon-workflow-run-success_svg__a)">
      <path
        fill="#3EC254"
        d="M.833 10A9.166 9.166 0 0 0 10 19.168a9.166 9.166 0 0 0 9.167-9.166A9.166 9.166 0 0 0 10 .834a9.166 9.166 0 0 0-9.167 9.167"
      ></path>
      <path
        fill="#fff"
        d="M6.077 9.755a.833.833 0 0 0 0 1.179l2.357 2.357a.833.833 0 0 0 1.179 0l4.714-4.714a.833.833 0 1 0-1.178-1.179l-4.125 4.125-1.768-1.768a.833.833 0 0 0-1.179 0"
      ></path>
    </g>
    <defs>
      <clipPath id="icon-workflow-run-success_svg__a">
        <path fill="#fff" d="M0 0h20v20H0z"></path>
      </clipPath>
    </defs>
  </svg>
);

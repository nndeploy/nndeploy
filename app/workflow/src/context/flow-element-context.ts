import React from 'react';

export const FlowElementContext = React.createContext<
  {
    element?: React.MutableRefObject<HTMLDivElement | null>;
  }

>({});


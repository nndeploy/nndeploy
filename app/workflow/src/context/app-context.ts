import React from "react";

export const AppContext = React.createContext<
  {
    nodeList?: React.MutableRefObject<HTMLDivElement | null>;

  }

>({});

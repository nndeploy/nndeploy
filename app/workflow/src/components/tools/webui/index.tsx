import { useReducer, useState } from 'react';

import { useClientContext } from '@flowgram.ai/free-layout-editor';
import { Button, SideSheet } from '@douyinfe/semi-ui';
import { useFlowEnviromentContext } from '../../../context/flow-enviroment-context';
import { FlowDocumentJSON } from '../../../typings';

import store, { initialState, reducer } from "./store/store";
import { initJson } from './store/actionType';
import WebUIComponents from './webUIComponents';

export function WebUI() {

  const { element: flowElementRef } = useFlowEnviromentContext()

  const clientContext = useClientContext();


  const [visible, setVisible] = useState(false);

  const [state, dispatch] = useReducer(reducer, (initialState))


  function onSure() {
    setVisible(false)

    clientContext.document.reload(state.json);

    setTimeout(() => {
      // 加载后触发画布的 fitview 让节点自动居中
      clientContext?.document.fitView();
    }, 100);
  }


  function onClose() {
    setVisible(false)

  }

  function getPopupContainer() {
    return flowElementRef?.current!
  }



  const onWebUi = () => {
    setVisible(true)
    const json: FlowDocumentJSON = clientContext.document.toJSON() as any
    dispatch(initJson(json))
  }


  return (
    <store.Provider value={{ state, dispatch }} >
      <Button
        //type="danger"

        onClick={onWebUi}
        //style={{ backgroundColor: 'rgba(255, 179, 171, 0.3)', borderRadius: '8px' }}
        style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
      >
        Webui
      </Button>
      <SideSheet
        width={"100%"}
        getPopupContainer={getPopupContainer}
        visible={visible}
        onCancel={onClose}
        title={<></>}
      >
        <WebUIComponents onSure={onSure} onClose={onClose} />
      </SideSheet>
    </store.Provider>

  );
}

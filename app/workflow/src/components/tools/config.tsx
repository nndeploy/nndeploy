import { useState, useEffect, useCallback } from 'react';

import { useClientContext, getNodeForm, FlowNodeEntity } from '@flowgram.ai/free-layout-editor';
import { Button, Badge } from '@douyinfe/semi-ui';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';

export function Config() {
  const [errorCount, setErrorCount] = useState(0);
  const clientContext = useClientContext();

  const flowEnviroment = useFlowEnviromentContext()



  /**
   * Validate all node and Save
   */
  const onConfig = useCallback(async () => {

    try {




      //const json = clientContext.document.toJSON()

      flowEnviroment.onConfig()

    } catch (e) {
      console.error('onSave', e)
    }

  }, [clientContext]);


  return (

    <Button
      //type="danger"

      onClick={onConfig}
      //style={{ backgroundColor: 'rgba(255, 179, 171, 0.3)', borderRadius: '8px' }}
       style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
    >
      Config
    </Button>

  );
}

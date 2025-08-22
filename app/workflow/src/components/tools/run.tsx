import { useState } from 'react';

import { getNodeForm, useClientContext, useService } from '@flowgram.ai/free-layout-editor';
import { Button } from '@douyinfe/semi-ui';

import { RunningService } from '../../services';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';

/**
 * Run the simulation and highlight the lines
 */
export function Run() {
  const [isRunning, setRunning] = useState(false);
  const runningService = useService(RunningService);

   const flowEnviroment = useFlowEnviromentContext()

   const {downloading} = flowEnviroment


   const clientContext = useClientContext();
   
  const onRun = async () => {
    setRunning(true);
    //await runningService.startRun();

     const allForms = clientContext.document.getAllNodes().map((node) => getNodeForm(node));

      await Promise.all(allForms.map(async (form) => form?.validate()));

      const json = clientContext.document.toJSON()
      flowEnviroment.onRun?.(json as any)

    setRunning(false);
  };
  return (
    <Button
      onClick={onRun}
      disabled = {downloading}
      loading={isRunning}
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
    >
      Run
    </Button>
  );
}

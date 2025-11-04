import { useState } from 'react';

import { getNodeForm, useClientContext, useService } from '@flowgram.ai/free-layout-editor';
import { Button, Toast } from '@douyinfe/semi-ui';

import { RunningService } from '../../services';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { apiWorkFlowRunStop } from '../../pages/Layout/Design/WorkFlow/api';

/**
 * Run the simulation and highlight the lines
 */
export function Run() {
  //const [isRunning, setRunning] = useState(false);

  const flowEnviroment = useFlowEnviromentContext()

  const { downloading,  runInfo, setRunInfo } = flowEnviroment

  const {runningTaskId, isRunning} = runInfo

  //console.log('isRunning', isRunning)


  const clientContext = useClientContext();

  const onRunOrStop = async () => {

    if (isRunning) {
        let resonse = await apiWorkFlowRunStop(runningTaskId)
        if(resonse.flag == 'success'){
          setRunInfo(oldRunInfo=>{
            return {
              ...oldRunInfo, 
              isRunning: false, 
              flowNodesRunningStatus: {}, 
              runningTaskId: '',
              result: '', 
            }
          })
          Toast.success('stop success')
        }
        //setRunning(false)
    } else {


      //setRunning(true);
      //await runningService.startRun();

      const allForms = clientContext.document.getAllNodes().map((node) => getNodeForm(node));

      await Promise.all(allForms.map(async (form) => form?.validate()));

      const json = clientContext.document.toJSON()
      flowEnviroment.onRun?.(json as any)

      //setRunning(false);
    }
  };
  return (
    <Button
      onClick={onRunOrStop}
      disabled={downloading}
      type= {isRunning ? 'danger': 'primary' }
      //loading={isRunning}
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
    >
      {isRunning ? 'Stop' : 'Run'}
    </Button>
  );
} 

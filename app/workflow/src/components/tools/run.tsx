import { useState } from 'react';

import { getNodeForm, useClientContext } from '@flowgram.ai/free-layout-editor';
import { Button, Toast } from '@douyinfe/semi-ui';

import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { apiWorkFlowRunStop } from '../../pages/Layout/Design/WorkFlow/api';
import { applyRunTaskState, applyRunSubmission } from './run-state';
import { retryRunTask } from './run-connection';

/**
 * Run the simulation and highlight the lines
 */
export function Run({ onRun }: { onRun?: () => void | Promise<void> } = {}) {
  const [stopping, setStopping] = useState(false);
  //const [isRunning, setRunning] = useState(false);

  const flowEnviroment = useFlowEnviromentContext()

  const { downloading,  runInfo, setRunInfo } = flowEnviroment

  const {runningTaskId, isRunning} = runInfo

  //console.log('isRunning', isRunning)


  const clientContext = useClientContext();

  const onRunOrStop = async () => {

    if (isRunning) {
        setStopping(true);
        try {
          if (runInfo.submissionPending) {
            const response = await retryRunTask(runningTaskId);
            setRunInfo(info => applyRunSubmission(info, runningTaskId, response.flag));
            if (response.message) Toast.info(response.message);
            return;
          }
          const response = await apiWorkFlowRunStop(runningTaskId);
          if (response.flag === 'success' && response.result) {
            setRunInfo(info => applyRunTaskState(info, response.result));
            Toast.info(response.message || 'Cancellation requested');
          } else {
            Toast.error(response.message || 'Unable to request cancellation');
          }
        } catch (error) {
          Toast.error(runInfo.submissionPending ? 'Unable to confirm submission. Checking task status.' : 'Unable to confirm cancellation. Please retry.');
        } finally {
          setStopping(false);
        }
        //setRunning(false)
    } else {
      if (onRun) {
        await onRun();
        return;
      }
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
      disabled={downloading || stopping || (isRunning && !runningTaskId)}
      type= {isRunning ? 'danger': 'primary' }
      //loading={isRunning}
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
    >
      {isRunning ? (runInfo.submissionPending ? 'Retry' : runInfo.cancelRequested ? 'Stopping…' : 'Stop') : 'Run'}
    </Button>
  );
} 

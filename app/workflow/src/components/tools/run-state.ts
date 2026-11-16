import type { IRunInfo } from '../../pages/components/flow/entity';

export interface RunTaskState {
  task_id: string;
  state: string;
  cancel_requested?: boolean;
  time_profile?: IRunInfo['log']['time_profile'];
  outputs?: Record<string, string>;
  outputs_truncated?: boolean;
  outputs_expired?: boolean;
  output_files?: IRunInfo['outputResource']['files'];
}

export function applyRunTaskState(info: IRunInfo, task: RunTaskState): IRunInfo {
  if (!task.task_id || info.runningTaskId !== task.task_id || !info.isRunning) return info;
  const result = task.state === 'SUCCEEDED' ? 'success'
    : task.state === 'FAILED' ? 'error' : task.state === 'CANCELLED' ? 'cancelled' : undefined;
  if (result) {
    return { ...info, isRunning: false, cancelRequested: false, submissionPending: false, result,
      flowNodesRunningStatus: task.time_profile?.nodes ?? info.flowNodesRunningStatus, time: Date.now(),
      ...(task.outputs || task.output_files ? { outputResource: { type: 'memory', task_id: task.task_id,
        content: task.outputs ?? {}, files: task.output_files, time: Date.now(), truncated: task.outputs_truncated, expired: task.outputs_expired } } : {}),
      ...(task.time_profile ? { log: { ...info.log, time_profile: task.time_profile } } : {}) };
  }
  if (['PENDING', 'DISPATCHED', 'RUNNING'].includes(task.state)) {
    if (task.cancel_requested || info.submissionPending) {
      return { ...info, submissionPending: false, cancelRequested: info.cancelRequested || !!task.cancel_requested };
    }
  }
  return info;
}

export function applyRunSubmission(info: IRunInfo, taskId: string, flag: string): IRunInfo {
  if (!info.isRunning || info.runningTaskId !== taskId || !info.submissionPending) return info;
  return { ...info, submissionPending: flag === 'pending' };
}

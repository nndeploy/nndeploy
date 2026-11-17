import React, { useState } from "react";
import store from "../store/store";
import lodash, { toNumber, trim } from 'lodash'
import IoType from "./ioType";
import { initJson } from "../store/actionType";
import styles from './index.module.scss'
import { getOutputFileLabel } from '../functions';
import { Button, Modal } from "@douyinfe/semi-ui";
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { Run } from '../../run';
import classNames from "classnames";
import UiParams from "./uiParams";
import CodeBlock from "../../../../pages/components/CodeBlock";
import { getResourceFields } from '../../../resource-files';
import { ResourceFieldsInput } from '../../../resource-files-input';
//import deployMarkDown from '../../../../assets/deploy.md'
//import deployMarkDown from '../../../../assets/deploy.md';
//import deployMarkDown from './deplolyContent'



interface IWebUIComponentsProps {
  onSure: () => void,
  onClose: () => void
}

const WebUIComponents: React.FC<IWebUIComponentsProps> = (props) => {


  const { state, dispatch } = React.useContext(store);

  const { json } = state;


  const flowEnviroment = useFlowEnviromentContext()

  const { runInfo } = flowEnviroment

  const { isRunning } = runInfo

  const [deployVisible, setDeployVisible] = useState(false);


  const showDeloyDialog = async function () {

    //const markdownContent = await import('../../../../assets/deploy.md?raw');
    setDeployVisible(true);
  };
  const handleDeployClose = () => {
    setDeployVisible(false);
    console.log('Cancel button clicked');
  };


  function getNodes(direction: 'Input' | 'Output', ioType: string[]) {
    return json?.nodes?.filter((node: any) => {

      return node.data.node_type_ === direction && ioType.includes(node.data.io_type_)
    })?.map(item => item.data).sort((left: any, right: any) => {
      const leftNameParts = left.name_.split('_')
      const leftNameNumberPart: string = leftNameParts[leftNameParts.length - 1]

      const rightNameParts = right.name_.split('_')
      const rightNameNumberPart: string = rightNameParts[rightNameParts.length - 1]


      if (isNumericWithTrim(leftNameNumberPart) && isNumericWithTrim(rightNameNumberPart)) {
        return lodash.parseInt(leftNameNumberPart) - lodash.parseInt(rightNameNumberPart)
      }
      else {
        return -1
      }
    })
  }

  function getAllUIParams() {

    let uiParams: string[] = []
    let cppUiParams: string[] = []
    let pythonUiParams: string[] = []

    for (const node of json.nodes) {

      let nodeUiParams: string[] = node.data.ui_params_ ? node.data.ui_params_ : []
      uiParams = [...uiParams, ...nodeUiParams]

      let nodeCppUiParams = node.data.param_?.['ui_params_'] ? node.data.param_?.['ui_params_'] : []
      cppUiParams = [...cppUiParams, ...nodeCppUiParams]

      let nodePythonUiParams = node.data.param?.['ui_params_'] ? node.data.param?.['ui_params_'] : []
      pythonUiParams = [...pythonUiParams, ...nodePythonUiParams]


    }
    return {
      uiParams,
      cppUiParams,
      pythonUiParams
    }
  }

  const allUiParams = getAllUIParams()

  const isNumericWithTrim = (str: string) => {
    const trimmed = trim(str);
    return trimmed !== '' && !isNaN(toNumber(trimmed));
  };


  // function getOutputNodes() {
  //   return json?.nodes?.filter((node: any) => node.data.node_type_ === 'Output')?.map(item => item.data).sort((left: any, right: any) => {
  //     const leftNameParts = left.name_.split('_')
  //     const leftNameNumberPart: string = leftNameParts[leftNameParts.length - 1]

  //     const rightNameParts = right.name_.split('_')
  //     const rightNameNumberPart: string = rightNameParts[rightNameParts.length - 1]


  //     if (isNumericWithTrim(leftNameNumberPart) && isNumericWithTrim(rightNameNumberPart)) {
  //       return lodash.parseInt(leftNameNumberPart) - lodash.parseInt(rightNameNumberPart)
  //     }
  //     else {
  //       return -1
  //     }
  //   })
  // }

  function getAllNodes() {
    return json?.nodes?.map(item => item.data).sort((left: any, right: any) => {
      const leftNameParts = left.name_.split('_')
      const leftNameNumberPart: string = leftNameParts[leftNameParts.length - 1]

      const rightNameParts = right.name_.split('_')
      const rightNameNumberPart: string = rightNameParts[rightNameParts.length - 1]


      if (isNumericWithTrim(leftNameNumberPart) && isNumericWithTrim(rightNameNumberPart)) {
        return lodash.parseInt(leftNameNumberPart) - lodash.parseInt(rightNameNumberPart)
      }
      else {
        return -1
      }
    })
  }

  const allNodes = getAllNodes()

  const resourceNodes = allNodes?.filter(node => getResourceFields(node).length > 0) || [];
  const inputBinaryNodes = getNodes('Input', ['Image', 'Video', 'Binary', 'Camera', 'Microphone', 'Model', 'Bool', 'Num'])
    ?.filter(node => getResourceFields(node).length === 0)
  const inputTextNodes = getNodes('Input', ['String', 'Text', 'Json', 'Xml', 'Csv', 'Yaml'])






  function onValueChange(value: any, node: any) {
    const newNodes = json.nodes.map(item => {
      if (item.data.name_ === node.name_) {

        return { ...item, data: { ...item.data, ...value } }
      } else {
        return item
      }
    })

    dispatch(initJson({ ...json, nodes: newNodes }))
  }



  return (
    <>
      <div className="drawer-content">
        <div className={styles['webui-components-container']}>
          <div className={styles['left-container']}>
            <h3>Inputs & parameters</h3>
            <p className={styles['section-description']}>Configure inputs, then run the workflow.</p>
            {resourceNodes.map(node => <div key={node.name_} className={styles['io-input-node-container']}>
              <div className={styles['io-node-name']}>{node.name_}</div>
              <ResourceFieldsInput node={node} onChange={value => onValueChange(value, node)} />
            </div>)}
            {
              inputBinaryNodes.length > 0 &&

              <div className={classNames(styles['io-input-area'], 'io-input-area')}>
                {inputBinaryNodes.map(node => {

                  return <div key={node.name_}
                    className={classNames(styles['io-input-node-container'], 'io-input-node-container')}>
                    <div className={styles['io-node-name']}>{node.name_}</div>
                    {
                      <IoType
                        direction={'input'}
                        ioDataType={node.io_type_}
                        node={node}

                        onChange={value => {
                          onValueChange(value, node)
                        }}
                      />
                    }
                  </div>

                })}
              </div>
            }
            {
              inputTextNodes.map(node => {

                return <div key={node.name_} className={classNames(styles['io-input-area'], styles['io-input-area-text'], 'io-input-area')}>

                  <div key={node.name_}
                    className={classNames(styles['io-input-node-container'], 'io-input-node-container')}>
                    <div className={styles['io-node-name']}>{node.name_}</div>
                    {
                      <IoType
                        direction={'input'}
                        ioDataType={node.io_type_}
                        node={node}

                        onChange={value => {
                          onValueChange(value, node)
                        }}
                      />
                    }
                  </div>
                </div>


              })}


            <div className={classNames(styles['param-area'], 'param-area')}>
              {
                allNodes.map(node => {
                  return <UiParams key={node.name_}
                    onChange={value => {
                      onValueChange(value, node)
                    }}
                    node={node}
                    allUiParams={allUiParams}
                    className={classNames(styles['ui-params'])} />
                })
              }

            </div>
          </div>
          <div className={styles['right-container']}>
            <h3>Results</h3>
            <p className={styles['section-description']} role="status">
              {isRunning ? 'Running workflow…' : runInfo.outputResource.expired
                ? 'Saved results have expired. Run the workflow again.'
                : runInfo.result === 'success' ? 'Workflow completed' : 'Run the workflow to view results.'}
            </p>
            {!isRunning && !runInfo.outputResource.expired && runInfo.result === 'success' && <>
              {(runInfo.outputResource.files || []).map(file => <article key={file.file_id} className={styles['result-card']}>
                <header><strong>{getOutputFileLabel(file)}</strong><span>{file.media_type}</span></header>
                {file.media_type.startsWith('image/') && <a href={file.url} target="_blank" rel="noreferrer"><img src={file.url} alt={getOutputFileLabel(file)} loading="lazy" /></a>}
                {file.media_type.startsWith('video/') && <video controls src={file.url} preload="metadata" />}
                {file.media_type.startsWith('audio/') && <audio controls src={file.url} preload="metadata" />}
                <a className={styles['download-link']} href={file.url} download>Download file</a>
              </article>)}
              {Object.entries(runInfo.outputResource.content || {}).map(([name, value]) =>
                <article key={name} className={styles['result-card']}>
                  <header><strong>{name}</strong><span>Text / data</span></header>
                  <pre>{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre>
                </article>)}
              {!runInfo.outputResource.files?.length && !Object.keys(runInfo.outputResource.content || {}).length &&
                <p>No output was returned by this workflow.</p>}
            </>}

          </div>
        </div>
      </div >
      <Modal
        title="Requirements"
        visible={deployVisible}
        onOk={handleDeployClose}
        afterClose={handleDeployClose} //>=1.16.0
        onCancel={handleDeployClose}
        closeOnEsc={true}
        width={600}
      >
        <div className={'tip-content'}>

          <CodeBlock
            code={'content.......'}
            language={'markdown'}
          />

        </div>

      </Modal>
      <div className="semi-sidesheet-footer">
        {/* <Button type="tertiary" onClick={() => showDeloyDialog()}>deploy</Button> */}
        <Button type="tertiary" onClick={() => props.onSure()}>confirm</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
        <Run onRun={async () => { await flowEnviroment.onRun?.(json as any); }} />
      </div>

    </>
  )
}

export default WebUIComponents

import classNames from "classnames";
import { IoTypeProps } from "../ioType/IoTypeImage"
import { useFlowEnviromentContext } from "../../../../../context/flow-enviroment-context";
import { PropertyEdit } from "../../../../../pages/components/flow/nodeRegistry/propertyEdit";
import { setNodeFieldValue } from "../../functions";
import styles from '../index.module.scss'
import React, { ReactNode } from "react";

interface IUiParamsProps {
  onChange: (node: any) => void;
  node: any,
  className: string, 
  allUiParams: {[paramType:string]: string[]}
}
const UiParams: React.FC<IUiParamsProps> = (props) => {

  const { onChange, node, className, allUiParams } = props;

  const registryKey = node.key_
  const nodeName = node.name_

  const { nodeList = [], paramTypes, } = useFlowEnviromentContext()

  let uiParams: string[] = node.ui_params_ ? node.ui_params_ : []
  let topParamContent = buildUiParams(uiParams, '')

  let cppUiParams = node.param_?.['ui_params_'] ? node.param_?.['ui_params_'] : []
  let cppParamsContent = buildUiParams(cppUiParams, 'param_')

  let pythonUiParams = node.param?.['ui_params_'] ? node.param?.['ui_params_'] : []
  let pythonParamsContent = buildUiParams(pythonUiParams, 'param')

  const hasUiParams = uiParams.length > 0 || cppUiParams.length > 0 || pythonUiParams.length > 0

  if(node.name_ == 'OpenCvImageEncode_23'){
    let j = 0
  }


  function isUIParamUniq(paramName: string, parent: string) {


    if(paramName == 'path_'){
      let j = 0
    }

    let paramType = parent == 'param_' ? 'cppUiParams' : parent == 'param' ? 'pythonUiParams' : 'uiParams'

    return allUiParams[paramType].filter(item=>item == paramName).length < 2
  }

  function buildUiParams(params: string[], parent: string) {

    if (params.length < 1) {
      return <></>
    }

    return <div className={classNames("property-container", className)}>
      <div className="UIProperties">
        {
          params.map((paramName, index) => {



            const fieldPaths = parent ? parent + '.' + paramName : paramName

            const value = parent ? node[parent][paramName] : node[paramName]
            const fieldNameLabel: ReactNode = isUIParamUniq(paramName, parent) 
            ? paramName :  
            <><span className={styles['nodeLael']}>[{nodeName}]</span>{paramName}</>

            return <PropertyEdit
              key={parent + paramName}
              fieldName={paramName}
              fieldNameLabel={fieldNameLabel}
              parentPaths={parent ? [parent] : []}

              value={value}

              onChange={(value) => {
                //field.onChange(value)
                const newNode = setNodeFieldValue(node, fieldPaths, value)

                onChange(newNode)
              }}
              onRemove={() => { }}
              onFieldRename={() => {

              }}

              showLine={false}

              //form={form}
              registryKey={registryKey}
              nodeList={nodeList}
              paramTypes={paramTypes}
              isLast={index == params.length - 1}
              topField={true}

            />
          })
        }
      </div>
    </div>
  }

  if (!hasUiParams) {
    return <></>
  }


  return (
    <div className={styles['ui-params-container']}>
      {
        (uiParams.length > 0) && <div className={styles['ui-params-group']}>
          {topParamContent}
        </div>

      }

      {
        (cppUiParams.length > 0) && <div className={styles['ui-params-group']}>
          <div className={styles['ui-params-title']}>param_</div>
          {cppParamsContent}
        </div>

      }
      {
        (pythonUiParams.length > 0) && <div className={styles['ui-params-group']}>
          <div className={styles['ui-params-title']}>param</div>
          {pythonParamsContent}
        </div>

      }

    </div>
  )
}
export default UiParams
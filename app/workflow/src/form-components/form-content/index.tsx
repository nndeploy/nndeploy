import React, { useEffect } from 'react';

import { Field, FieldArray, FlowNodeRegistry, useWatchFormValueIn, WorkflowNodePortsData, WorkflowPorts } from '@flowgram.ai/free-layout-editor';

import { useIsSidebar, useNodeRenderContext } from '../../hooks';
import { FormTitleDescription, FormWrapper } from './styles';
import { Tooltip } from '@douyinfe/semi-ui';

/**
 * @param props
 * @constructor
 */
export function FormContent(props: { children?: React.ReactNode }) {
  const { node, expanded, form } = useNodeRenderContext();
  const isSidebar = useIsSidebar();
  const registry = node.getNodeRegistry<FlowNodeRegistry>();

  const isContainer = form?.getValueIn('is_graph_')

  // useEffect(() => {


  //   if(form?.getValueIn('name_') === 'Prefill_1'){
  //     let i = 0
  //   }
  //   const inputs: any[] = form?.getValueIn('inputs_') || []

  //   const inputPorts: WorkflowPorts = inputs.map(input => ({
  //     type: 'input',
  //     portID: input.id,
  //     location: 'left',

  //   }));

  //   const outputs: any[] = form?.getValueIn('outputs_') || []

  //   const outputPorts: WorkflowPorts = outputs.map(output => ({
  //     type: 'output',
  //     portID: output.id,
  //     location: 'right',

  //   }));

  //   const allPorts = [...inputPorts, ...outputPorts]

  //   node.getData(WorkflowNodePortsData).updateDynamicPorts(allPorts)

  //   let j =0
  //   j= j+ 1

  // }, [
  //   form?.getValueIn('inputs_'),
  //   form?.getValueIn('outputs_')
  // ])

 
  return (
    <FormWrapper className='form-content'>

      {!isSidebar && <div className="connection-area" >
        {
          isContainer && expanded ? <></> :
            <>
              <div className="input-area">
                <FieldArray name="inputs_">
                  {({ field }) => {

                      let temp = field.map
                      if(!temp){
                        debugger
                         let  name_ = form?.getValueIn('name_')
                         let j = 0 

                      }

                    return <>
                      {field.map((child, index) => {
                        return (
                          <Field<any> key={child.name} name={child.name}>
                            {({ field: childField, fieldState: childState }) => {

                              const truncatedValueType = childField.value.type_?.length > 12 ? childField.value.type_.substring(0, 12) + '...' : childField.value.type_
                              //Playgroundconsole.log('inputs_ childField.value.id', childField.value.id)
                              return (
                                <div
                                  style={{
                                    fontSize: 12,
                                    marginBottom: 6,
                                    width: '100%',
                                    position: 'relative',
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    gap: 8,
                                  }}
                                >
                                  <div
                                    style={{
                                      justifyContent: 'start',
                                      alignItems: 'center',
                                      color: 'var(--semi-color-text-0)',
                                      width: 118,
                                      position: 'relative',
                                      display: 'flex',
                                      columnGap: 4,
                                      flexShrink: 0,
                                    }}
                                  >

                                    <Tooltip content={<>
                                      <p>type: {childField.value.type_}</p>
                                      <p>desc: {childField.value.desc_}</p>
                                    </>}>{truncatedValueType}</Tooltip>
                                  </div>

                                  <div
                                    style={{
                                      flexGrow: 1,
                                      minWidth: 0,
                                    }}
                                  >
                                    <div
                                      className="connection-point connection-point-left"
                                      data-port-id={childField.value.id}
                                      data-port-type="input"
                                      data-port-location="left"
                                      data-port-desc={childField.value.desc_}
                                    //data-port-wangba={childField.value.desc_}
                                    ></div>
                                  </div>
                                </div>
                              );
                            }}
                          </Field>
                        );
                      })}
                    </>
                  }}
                </FieldArray>

              </div>
              <div className="output-area">
                <FieldArray name="outputs_">
                  {({ field }) => {

                    return <>
                      {field.map((child, index) => (
                        <Field<any> key={child.name} name={child.name}>
                          {({ field: childField, fieldState: childState }) => {

                            const truncatedValueType = childField.value.type_?.length > 12 ? childField.value.type_.substring(0, 12) + '...' : childField.value.type_
                            return <>

                              <div
                                style={{
                                  fontSize: 12,
                                  marginBottom: 6,
                                  width: '100%',
                                  position: 'relative',
                                  display: 'flex',
                                  justifyContent: 'flex-end',
                                  alignItems: 'center',
                                  gap: 8,
                                }}
                              >
                                <div
                                  style={{
                                    justifyContent: 'end',
                                    alignItems: 'flex-end',
                                    color: 'var(--semi-color-text-0)',
                                    width: 118,
                                    position: 'relative',
                                    display: 'flex',
                                    columnGap: 4,
                                    flexShrink: 0,
                                  }}
                                >

                                  <Tooltip content={<>
                                    <p>type: {childField.value.type_}</p>
                                    <p>desc: {childField.value.desc_}</p>
                                  </>}>{truncatedValueType}</Tooltip>
                                </div>

                                <div
                                  style={{
                                    //flexGrow: 1,
                                    minWidth: 0,
                                  }}
                                >

                                  <div
                                    className="connection-point connection-point-right"
                                    data-port-location="right"
                                    data-port-id={childField.value.id}
                                    data-port-type="output"
                                    data-port-desc={childField.value.desc_}
                                  >

                                  </div>
                                </div>
                              </div>



                            </>
                          }}
                        </Field>
                      ))}
                    </>
                  }}
                </FieldArray>
              </div>
            </>
        }

      </div>
      }
      {expanded  || isSidebar? (
        <>
          {/* {isSidebar && <FormTitleDescription>{registry.info?.description}</FormTitleDescription>} */}
          {props.children}
        </>
      ) : undefined}
    </FormWrapper>
  );
}

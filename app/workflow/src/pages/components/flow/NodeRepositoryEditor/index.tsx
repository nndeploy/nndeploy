import React, { useState, useEffect, useRef } from 'react';
import { Button, Select, SideSheet, Switch, Typography } from '@douyinfe/semi-ui';

import { Form, Field, FieldArray, useForm } from "@flowgram.ai/free-layout-editor";
import { INodeEntity } from '../../../Node/entity';
import { createForm } from '@flowgram.ai/form'

import lodash, { random } from "lodash";
import { IParamTypes } from '../../../Layout/Design/WorkFlow/entity';
import { getFieldType } from '../functions';
import { FormItem } from '../form-item';
import { FxExpression } from '../../../../form-components/fx-expression';
import { Feedback } from '../../../../form-components';
import Section from '@douyinfe/semi-ui/lib/es/form/section';
import { FormParams } from '../form-params';
import { IconCrossCircleStroked, IconPlus } from '@douyinfe/semi-icons';
import { useIsSidebar } from '../../../../hooks';
import { PropertyEdit } from '../nodeRegistry/propertyEdit';

const { Text } = Typography;

interface NodeEntityFormProps {
  
  nodeEntity: INodeEntity;

  nodeList: INodeEntity[],
  paramTypes: IParamTypes,

  visible: boolean;
  onClose: () => void;
  onSave: (updatedNode: INodeEntity) => void;
}

/**
 * 单个节点的独立表单，支持编辑当前节点及递归编辑子节点
 * 保存子节点时，更新当前表单对应字段
 */
export const NodeEntityForm: React.FC<NodeEntityFormProps> = (props) => {

  const { nodeEntity, visible, onClose, onSave, paramTypes, nodeList } = props

  const registryKey = nodeEntity.key_;

  //const readonly = !useIsSidebar();

  const formRef = useRef<any>()

  const [childEditingNode, setChildEditingNode] = useState<INodeEntity | null>(null);
  const [childDrawerVisible, setChildDrawerVisible] = useState(false);

  const openChildEditor = (childNode: INodeEntity) => {
    setChildEditingNode(childNode);
    setChildDrawerVisible(true);
  };

  const closeChildEditor = () => {
    setChildEditingNode(null);
    setChildDrawerVisible(false);
  };

  /**
   * 子节点保存后，更新当前Form中对应的node_repository_字段
   */
  const handleChildSave = (updatedChild: INodeEntity) => {
    const currentValues = formRef.current.values
    const updatedChildren = (currentValues.node_repository_ || []).map((child: INodeEntity) =>
      child.key_ === updatedChild.key_ ? updatedChild : child
    );
    formRef.current.setValueIn('node_repository_', updatedChildren);
    closeChildEditor();
  };

  const handleFinish = () => {

    formRef.current.validateFields().then((values: INodeEntity) => {
      onSave(values);
      onClose();
    });

  };

  const excludeFields = [
    "key_",
    "is_graph_",
    "is_inner_",
    "node_type_",
    "is_time_profile_",
    "is_debug_",
    "is_external_stream_",
    // "param_",
    // "inputs_",
    // "outputs_",
    // "node_repository_",
    'is_dynamic_input_',
    'is_dynamic_output_'
  ];
  const basicFields = lodash.difference(
    Object.keys(nodeEntity),
    excludeFields
  );

  function onDrawerClose() {

    let formValues = formRef.current.values



    onSave(formValues);
    //onClose(formValues)
  }

  return (
    <>
      <SideSheet visible={visible} onCancel={onDrawerClose} title={`[edit] ${nodeEntity.name_}`} width={500}>

        <Form initialValues={nodeEntity}>
          {({ form }) => {

            formRef.current = form;

             

            return <form
              onSubmit={e => {
                e.preventDefault();
                // 调用 form 的提交方法或验证方法
                //form.submit();

              }}
            >
              <div className="property-container">
                <div className="UIProperties">
                  {basicFields.map((fieldName, index) => {
                    //const fieldType = getFieldType([fieldName], form, nodeList, paramTypes)

                    return (
                      <Field key={fieldName} name={fieldName} >
                        {({ field, fieldState }) => {
                          return <PropertyEdit fieldName={fieldName} parentPaths={[]}
                            // value={form.getValueIn(fieldName)}

                            // onChange={(value) => {
                            //   form.setValueIn(fieldName, value)
                            // }}
                            fieldNameLabel={fieldName}
                            value={field.value}

                            onChange={(value) => {
                              field.onChange(value)
                            }}
                            onRemove={() => { }}
                            onFieldRename={() => {

                            }}

                            showLine={false}
                            topField={true}

                            //form={form}
                            registryKey={registryKey}
                            nodeList={nodeList}
                            paramTypes={paramTypes}
                            isLast={index == basicFields.length - 1}

                          />
                        }}
                      </Field>

                    )

                    // if (fieldType.isArray) {
                    //   return (
                    //     <div className="number-array-field">
                    //       <div className="field-label">{fieldName}</div>
                    //       <div className="filed-array-items">
                    //         <FieldArray name={`${fieldName}`}>
                    //           {({ field }) => {

                    //             if (!lodash.isFunction(field.map)) {
                    //               var xx = 0
                    //             }
                    //             return <>
                    //               {field.map((child, index) => (
                    //                 <Field key={child.name} name={child.name}>
                    //                   {({
                    //                     field: childField,
                    //                     fieldState: childState,
                    //                   }) => (
                    //                     <div className="expression-field" style={{ width: '100%' }}
                    //                     >
                    //                       <>
                    //                         {

                    //                           fieldType.componentType == 'boolean' ?
                    //                             <Switch checked={!!childField.value}
                    //                               //label='开关(Switch)' 
                    //                               onChange={(value: boolean) => {
                    //                                 childField.onChange(value)
                    //                               }} />
                    //                             : fieldType.componentType == 'select' ?
                    //                               <Select

                    //                                 value={childField.value as number}
                    //                                 style={{ width: '100%' }}
                    //                                 optionList={paramTypes[fieldType.selectKey!].map(item => {
                    //                                   return {
                    //                                     label: item,
                    //                                     value: item
                    //                                   }
                    //                                 })}

                    //                                 onChange={(value) => {
                    //                                   childField.onChange(value)
                    //                                 }}

                    //                               >

                    //                               </Select> :

                    //                               <FxExpression
                    //                                 value={childField.value as number}
                    //                                 fieldType={fieldType}
                    //                                 onChange={(v) => childField.onChange(v)}
                    //                                 icon={
                    //                                   <Button
                    //                                     theme="borderless"
                    //                                     icon={<IconCrossCircleStroked />}
                    //                                     onClick={() => field.delete(index)}
                    //                                   />
                    //                                 }
                    //                                 hasError={
                    //                                   Object.keys(childState?.errors || {})
                    //                                     .length > 0
                    //                                 }
                    //                                 readonly={false}
                    //                               />


                    //                         }
                    //                         <Feedback
                    //                           errors={childState?.errors}
                    //                           invalid={childState?.invalid}
                    //                         />
                    //                       </>
                    //                     </div>
                    //                   )}
                    //                 </Field>
                    //               ))}
                    //               {!false && (
                    //                 <div>
                    //                   <Button
                    //                     theme="borderless"
                    //                     icon={<IconPlus />}
                    //                     onClick={() => field.append(0)}
                    //                   >
                    //                     Add
                    //                   </Button>
                    //                 </div>
                    //               )}
                    //             </>
                    //           }}
                    //         </FieldArray>
                    //       </div>
                    //     </div>
                    //   );
                    // }
                    // return (
                    //   <Field key={fieldName} name={fieldName}>
                    //     {({ field, fieldState }) => {
                    //       if (fieldName == 'flag_') {
                    //         //debugger
                    //         let i = 0
                    //       }
                    //       if (fieldName == 'image_url_') {
                    //         let j = 0
                    //       }


                    //       return <FormItem
                    //         name={fieldName}
                    //         type={"string" as string}
                    //         required={true}
                    //       >

                    //         <div className="expression-field"
                    //         >
                    //           <>
                    //             {

                    //               fieldType.componentType == 'boolean' ?
                    //                 <Switch checked={!!field.value}
                    //                   //label='开关(Switch)' 
                    //                   onChange={(value: boolean) => {
                    //                     field.onChange(value)
                    //                   }} />
                    //                 : fieldType.componentType == 'select' ?
                    //                   <Select
                    //                     value={field.value as string}
                    //                     style={{ width: '100%' }}
                    //                     onChange={(value) => {
                    //                       field.onChange(value)
                    //                     }
                    //                     }


                    //                     optionList={paramTypes[fieldType.selectKey!].map(item => {
                    //                       return {
                    //                         label: item,
                    //                         value: item
                    //                       }
                    //                     })}>

                    //                   </Select> :

                    //                   <FxExpression
                    //                     value={field.value as string}
                    //                     fieldType={fieldType}
                    //                     onChange={field.onChange}
                    //                     readonly={false}
                    //                     hasError={
                    //                       Object.keys(fieldState?.errors || {}).length > 0
                    //                     }
                    //                     icon={<></>}
                    //                   />


                    //             }
                    //             <Feedback
                    //               errors={fieldState?.errors}
                    //               invalid={fieldState?.invalid}
                    //             />
                    //           </>
                    //         </div>
                    //       </FormItem>
                    //     }}
                    //   </Field>
                    // );


                  })}
                </div>
              </div>

              {/* {
                nodeEntity.hasOwnProperty('param_') && <Section text={"param_"}>
                  <FormParams />
                </Section>
              }
              {
  

                <Field<any> name="node_repository_">
                  {({ field: node_repository_ }) => {

                    if (node_repository_.value && node_repository_.value.length > 0) {
                      return node_repository_.value.map((child: INodeEntity) => {
                        return <div key={child.key_} style={{ marginLeft: 20, marginBottom: 8 }}>
                          <Text link onClick={() => openChildEditor(child)}>{child.name_}</Text>

                        </div>
                      })

                    } else {
                      return <></>
                    }

                  }}
                </Field>
              } */}



            </form>
          }
          }
        </Form>

      </SideSheet>

      {childEditingNode && (
        <NodeEntityForm
          nodeEntity={childEditingNode}
          nodeList={nodeList}
          paramTypes={paramTypes}
        
          visible={childDrawerVisible}
          onClose={closeChildEditor}
          onSave={handleChildSave}
        />
      )}
    </>
  );
};

interface NodeRepositoryEditorProps {
  registryKey: string;
  node_repository_?: INodeEntity[];
  nodeList: INodeEntity[],
  paramTypes: IParamTypes,
  onUpdate: (updatedList: INodeEntity[]) => void;
}

/**
 * 递归更新node_repository_中被编辑的节点
 */
const updateNodeInList = (list: INodeEntity[], updatedNode: INodeEntity): INodeEntity[] => {
  return list.map(node => {
    if (node.name_ === updatedNode.name_) {
      return updatedNode;
    }
    if (node.node_repository_ && node.node_repository_.length > 0) {
      return {
        ...node,
        node_repository_: updateNodeInList(node.node_repository_, updatedNode),
      };
    }
    return node;
  });
};

/**
 * 顶层组件，渲染node_repository_列表，点击编辑打开顶层NodeEntityForm
 */
const NodeRepositoryEditor: React.FC<NodeRepositoryEditorProps> = (props) => {

  const { nodeList, paramTypes, node_repository_, onUpdate, registryKey } = props

  const [selectedNode, setSelectedNode] = useState<INodeEntity | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);

  const openEditor = (node: INodeEntity) => {
    setSelectedNode(node);
    setDrawerVisible(true);
  };

  const closeEditor = () => {
    setSelectedNode(null);
    setDrawerVisible(false);
  };

  const handleSave = (updatedNode: INodeEntity) => {
    if (!node_repository_) return;

    closeEditor()
    const updatedList = updateNodeInList(node_repository_, updatedNode);
    onUpdate(updatedList);
  };

  return (
    <div>
      {node_repository_ && node_repository_.length > 0 ? (
        node_repository_.map(node => (
          <div key={node.key_} style={{ marginBottom: 8 }}>
            <Text link onClick={() => openEditor(node)}>{node.name_}</Text>

          </div>
        ))
      ) : <></>}

      {selectedNode && (
        <NodeEntityForm
          nodeEntity={selectedNode}
          visible={drawerVisible}
          onClose={closeEditor}
          onSave={handleSave}
          nodeList={nodeList}
          paramTypes={paramTypes}
         
        />
      )}
    </div>
  );
};

export default NodeRepositoryEditor;

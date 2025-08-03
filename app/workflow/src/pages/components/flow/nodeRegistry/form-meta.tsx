import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldArray,
  useFieldValidate,
  useWatch,
  useWatchFormValueIn,
  useNodeRender,
  useWatchFormValues,
} from "@flowgram.ai/free-layout-editor";
import { Modal, Select, SideSheet, Switch, TextArea, Typography, VideoPlayer } from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { Feedback, FormContent } from "../../../../form-components";
import { useIsSidebar } from "../../../../hooks";

import "./index.scss";
import { FormHeader } from "../form-header";
import { FormItem } from "../form-item";
import { FormParams } from "../form-params";
import { FxExpression } from "../../../../form-components/fx-expression";
import Section from "@douyinfe/semi-ui/lib/es/form/section";
import { GroupNodeRender } from "../../../../components";
import lodash, { random, uniqueId } from "lodash";
import { FormDynamicPorts } from "../form-dynamic-ports";
import { FlowEnviromentContext, useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { getFieldType } from "../functions";
import { INodeEntity } from "../../../Node/entity";
import { useContext, useEffect, useRef, useState } from "react";
import RepositoryItemDrawer from "../NodeRepositoryEditor";
import NodeRepositoryEditor from "../NodeRepositoryEditor";
import { IResourceTreeNodeEntity } from "../../../Layout/Design/Resource/entity";
import ResourceEditDrawer from "../../../Layout/Design/Resource/ResourceEditDrawer";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();

  // const { node } = useNodeRender();

  const { nodeList = [], paramTypes, element: flowElementRef, outputResources } = useFlowEnviromentContext()



  //console.log("form.values", form.values);

  // const basicFields = ["name_", "device_type_", "type_"].filter((item) =>
  //   form.values.hasOwnProperty(item)
  // );

  const is_dynamic_input_ = form.getValueIn("is_dynamic_input_");
  const is_dynamic_output_ = form.getValueIn("is_dynamic_output_");

  const path_ = form.getValueIn("path_") as string;

  const key_ = form.getValueIn("key_");


  const excludeFields = [
    "id",
    "key_",
    "param_",
    "inputs_",
    "outputs_",
    "node_repository_",
    'is_dynamic_input_',
    'is_dynamic_output_',
    "is_graph_",
    "is_inner_",
    "node_type_"

  ];
  const basicFields = lodash.difference(
    Object.keys(form.values),
    excludeFields
  );

  const [respository, setRespository] = useState<INodeEntity>({} as INodeEntity)
  const [repositoryDrawerVisible, setRepositoryDrawerVisible] = useState(false)

  function onShowRepositoryItemDrawer(respository: INodeEntity) {
    setRespository(respository)
    setRepositoryDrawerVisible(true)
  }

  function onRepositoryDrawerClose() {
    setRepositoryDrawerVisible(false)
  }

  function onRepositoryDrawerSave(respository: INodeEntity) {

  }

  function isMediaNode() {

    const imageNodes: string[] = ['nndeploy::codec::OpenCvImageDecode', 'nndeploy::codec::OpenCvImageEncode']
    if (imageNodes.includes(key_)) {
      return true
    }
    return false
  }

  function isTextNode() {
    const textNodes: string[] = ['nndeploy::qwen::PrintNode',]
    if (textNodes.includes(key_)) {
      return true
    }
    return false
  }

  function needShowTextContent() {

    const nodeNames = outputResources.text.map(item => item.name)

    const needShow = isTextNode() && nodeNames.includes(form.getValueIn('name_')) //path_.includes('&time=')

    return needShow
  }

  function isInputMediaNode() {
    const imageNodes: string[] = ['nndeploy::codec::OpenCvImageDecode', 'nndeploy::codec::OpenCvVideoDecode']
    if (imageNodes.includes(key_)) {
      return true
    }
    return false
  }

  function isOutputMediaNode() {
    const imageNodes: string[] = ['nndeploy::codec::OpenCvImageEncode', 'nndeploy::codec::OpenCvVideoEncode']
    if (imageNodes.includes(key_)) {
      return true
    }
    return false
  }


  function isImageFile(filename: string) {
    // 支持的图片后缀（不区分大小写）
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff'];
    const lowerFilename = filename.toLowerCase();
    return imageExtensions.some(ext => lowerFilename.endsWith(ext));
  }
  // 新增判断视频文件的函数
  function isVideoFile(filename: string) {
    // 支持的视频后缀（不区分大小写）
    const videoExtensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.mpeg', '.mpg'];
    const lowerFilename = filename.toLowerCase();
    return videoExtensions.some(ext => lowerFilename.endsWith(ext));
  }

  const [resourceEdit, setResourceEdit] = useState<IResourceTreeNodeEntity>();
  const [resoureEditVisible, setResoureEditVisible] = useState(false)


  function onShowMediaFile(event: React.MouseEvent<HTMLElement, MouseEvent>, file: string) {
    event.stopPropagation();
    event.preventDefault();

    //setFile(file)
    const parentId = file.includes('images') ? 'images' : file.includes('videos') ? 'videos' : ''
    const fileName = file.substring(file.lastIndexOf('/') + 1)
    setResourceEdit({ id: uniqueId(), parentId, type: 'leaf', name: fileName })
    setResoureEditVisible(true)

  }


  function handleResoureDrawerClose() {
    setResoureEditVisible(false)
  }
  function onResourceEditDrawerClose() {
    setResoureEditVisible(false)
  }
  function onResourceEditDrawerSure() {
    setResoureEditVisible(false)
  }

  const renderFormRef = useRef<any>();

  // const getContainer = () => {
  //       return document.querySelector('#root')! as HTMLDivElement;
  //   };

  function getPopupContainer() {

    let container = renderFormRef?.current;
    while (container && !(container instanceof HTMLElement && container.classList.contains('demo-container'))) {
      container = container.parentElement;
    }
    return container

    // const container = document.querySelector('div.demo-container')! as HTMLDivElement
    // const container = document.querySelector('#root')! as HTMLDivElement  //div.demo-container
    //  return container
  }
  function needShowMedia() {

    const nodeNames = outputResources.path.map(item => item.name)
    const needShow = isInputMediaNode() || (isOutputMediaNode() && nodeNames.includes(form.getValueIn('name_')))  //path_.includes('&time=')

    return needShow
  }

  const [updateVal, setUpdateVal] = useState({})

  // const update = ()=>{
  //   setUpdateVal({})
  // }

  // useEffect(()=>{

  // }, [updateVal])

  useEffect(() => {
    if (outputResources) {
      console.log('outputResources', outputResources,)
      console.log('name', form.getValueIn('name_'))
    }


  }, [outputResources])

  return (
    <div className="drawer-render-form" ref={renderFormRef}>
      <FormHeader />

      <FormContent>
        {!isSidebar && (
          <>
            <div className="connection-area">
              <div className="input-area">
                <FieldArray name="inputs_">
                  {({ field }) => (
                    <>
                      {field.map((child, index) => {
                        return (
                          <Field<any> key={child.name} name={child.name}>
                            {({ field: childField, fieldState: childState }) => {
                              return (
                                <FormItem
                                  name={`${childField.value.type_}`}///${childField.value.desc_}
                                  type="boolean"
                                  required={false}
                                //labelWidth={40}
                                >
                                  <div
                                    className="connection-point connection-point-left"
                                    data-port-id={childField.value.id}
                                    data-port-type="input"
                                    data-port-desc={childField.value.desc_}
                                  //data-port-wangba={childField.value.desc_}
                                  ></div>
                                </FormItem>
                              );
                            }}
                          </Field>
                        );
                      })}
                    </>
                  )}
                </FieldArray>
                {/* {
              inputValues.map((item) => {
                
                return (
                 
                  <FormItem
                   key={`${item.type_}/${item.desc_}`}
                    name={`${item.type_}/${item.desc_}`}
                    type="boolean"
                    required={false}
                    //labelWidth={40}
                  >
                    <div
                      className="connection-point connection-point-left"
                      data-port-id={item.desc_}
                      data-port-type="input"
                    ></div>
                  </FormItem>
                )
              }
              )
            } */}
              </div>
              <div className="output-area">
                <FieldArray name="outputs_">
                  {({ field }) => (
                    <>
                      {field.map((child, index) => (
                        <Field<any> key={child.name} name={child.name}>
                          {({ field: childField, fieldState: childState }) => (
                            <FormItem
                              name={`${childField.value.type_}`}///${childField.value.desc_}
                              type="boolean"
                              required={false}
                            //labelWidth={40}
                            >
                              <div
                                className="connection-point connection-point-right"
                                data-port-id={childField.value.id}
                                data-port-type="output"
                                data-port-desc={childField.value.desc_}
                              ></div>
                            </FormItem>
                          )}
                        </Field>
                      ))}
                    </>
                  )}
                </FieldArray>
              </div>
            </div>

            <Field key={'path_'} name={'path_'}>
              {({ field, fieldState }) => {

                if (needShowMedia() && field.value && isImageFile(field.value as string)) {

                  const url = `/api/preview?file_path=${field.value}&time=${Date.now()}`
                  return <div className="resource-preview">
                    <img src={url} onClick={((event) => onShowMediaFile(event, field.value as string))}

                    />
                  </div>
                } else if (needShowMedia() && field.value && isVideoFile(field.value as string)) {

                  const url = `/api/preview?file_path=${field.value}&time=${Date.now()}`
                  return <div className="resource-preview"
                    onClick={((event) => onShowMediaFile(event, field.value as string))}
                  >
                    <VideoPlayer
                      height={100}
                      controlsList={[]}
                      clickToPlay={false}
                      autoPlay={true}

                      src={`/api/preview?file_path=${field.value}&time=${new Date().getTime()}`}
                    //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
                    />


                  </div>
                } else {
                  return <></>
                }
              }}

            </Field>
            {
              isTextNode() && needShowTextContent() &&
              <TextArea rows={8} value={outputResources.text.find(item => item.name == form.getValueIn('name_'))?.text}>

              </TextArea>
            }

          </>
        )}
        {isSidebar ? (
          <>
            {/* <Section text={"basic"}> */}
            {basicFields.map((fieldName) => {
              return (
                <Field key={fieldName} name={fieldName}>
                  {({ field, fieldState }) => {
                    if (fieldName == 'flag_') {
                      //debugger
                      let i = 0
                    }
                    const fieldType = getFieldType([fieldName], form, nodeList, paramTypes)
                    return <FormItem
                      name={fieldName}
                      type={"string" as string}
                      required={true}
                    >

                      <div className="expression-field"
                      >
                        <>
                          {

                            fieldType.componentType == 'boolean' ?
                              <Switch checked={!!field.value}
                                //label='开关(Switch)' 
                                onChange={(value: boolean) => {
                                  field.onChange(value)
                                }} />
                              : fieldType.componentType == 'select' ?
                                <Select
                                  value={field.value as string}
                                  style={{ width: '100%' }}
                                  onChange={(value) => {
                                    field.onChange(value)
                                  }
                                  }


                                  optionList={paramTypes[fieldType.selectKey!].map(item => {
                                    return {
                                      label: item,
                                      value: item
                                    }
                                  })}>

                                </Select> :

                                <FxExpression
                                  value={field.value as string}
                                  fieldType={fieldType}
                                  onChange={(value) => {
                                    field.onChange(value)
                                    //setUpdateVal({})
                                  }
                                  }
                                  readonly={!isSidebar || fieldName == 'desc_'}
                                  hasError={
                                    Object.keys(fieldState?.errors || {}).length > 0
                                  }
                                  icon={<></>}
                                />


                          }
                          <Feedback
                            errors={fieldState?.errors}
                            invalid={fieldState?.invalid}
                          />
                        </>
                      </div>
                    </FormItem>
                  }}
                </Field>
              );
            })}
            {/* </Section> */}

            {is_dynamic_input_ && (
              <Section text={"inputs_"}>
                <FormDynamicPorts portType="inputs_" />
              </Section>
            )}

            {is_dynamic_output_ && (
              <Section text={"outputs_"}>
                <FormDynamicPorts portType="outputs_" />
              </Section>
            )}
            {
              form.values.hasOwnProperty('param_') && <Section text={"param_"}>
                <FormParams />
              </Section>
            }

            {
              // form.values.hasOwnProperty('node_repository_') && <Section text={"node_repository_"}>
              //   {

              //     <NodeRepositoryEditor node_repository_ = {form.getValueIn('node_repository_') as INodeEntity[]}
              //       nodeList = {nodeList} paramTypes = {paramTypes}


              //     onUpdate={(values)=>{
              //        //console.log(values)
              //        form.setValueIn('node_repository_', values)

              //        var temp = form.values
              //        var i = 0
              //     }}  />
              //   }
              // </Section>

              <Field<any> name="node_repository_">
                {({ field: node_repository_ }) => {

                  if (node_repository_.value && node_repository_.value.length > 0) {
                    return <Section text={"node_repository_"}>
                      {

                        <NodeRepositoryEditor node_repository_={form.getValueIn('node_repository_') as INodeEntity[]}
                          nodeList={nodeList} paramTypes={paramTypes}


                          onUpdate={(values) => {
                            //console.log(values)
                            node_repository_.onChange(values)
                          }} />
                      }
                    </Section>
                  } else {
                    return <></>
                  }

                }}
              </Field>

              
            }
             {
              isTextNode() && needShowTextContent() &&
              <TextArea rows={8}  value={outputResources.text.find(item => item.name == form.getValueIn('name_'))?.text}>

              </TextArea>
            }

          </>
        ) : (
          <></>
        )}
      </FormContent>
      {/* <SideSheet title="滑动侧边栏" visible={repositoryDrawerVisible} onCancel={onRepositoryDrawerClose}>
                <RepositoryItemDrawer respository = {respository} onRepositoryDrawerSave = {onRepositoryDrawerSave}/>

            </SideSheet> */}
      {/* 
      <Modal
        title="file preview"
        visible={fileModelVisible}
        //onOk={handleOk}
        //afterClose={fileModelClose} //>=1.16.0
       // onCancel={fileModelClose}
        closeOnEsc={false}
        //zIndex={100000000000000}
      >
        <>
        <h2>model content..........</h2>
          {file?.includes("images") ?
            <div className="image-preview">
              <img src={`/api/preview/images/${file.split('images')[1]}`} />
            </div>
            : file.includes("videos") ?
              <div className="video-preview">
                <VideoPlayer
                  height={430}
                  src={`/api/preview/videos/${file.split('images')[1]}}`}
               
                />
              </div>
              : <></>

          }
        </>
      </Modal> */}
      <SideSheet
        width={"60%"}
        mask={true}
        visible={resoureEditVisible}
        onCancel={handleResoureDrawerClose}
        closeOnEsc={true}
        title={'resource preview'}

        //zIndex={10000}
        getPopupContainer={getPopupContainer}
      >
        <ResourceEditDrawer
          node={resourceEdit!}
          onSure={onResourceEditDrawerSure}
          onClose={onResourceEditDrawerClose}
          showFileInfo={false}
        />

      </SideSheet>
    </div >
  );
};

export const formMeta: FormMeta<FlowNodeJSON> = {
  render: renderForm,
  validateTrigger: ValidateTrigger.onChange,
  validate: {
    // title: ({ value }: { value: string }) =>
    //   value ? undefined : "Title is required",
    // "inputsValues.conditions.*": ({ value }) => {
    //   if (!value?.value?.content) return "Condition is required";
    //   return undefined;
    // },
  },
};

// export const groupFormMeta: FormMeta<FlowNodeJSON> = {
//   render: GroupNodeRender,
//   validateTrigger: ValidateTrigger.onChange,
//   validate: {
//     title: ({ value }: { value: string }) =>
//       value ? undefined : "Title is required",
//     "inputsValues.conditions.*": ({ value }) => {
//       if (!value?.value?.content) return "Condition is required";
//       return undefined;
//     },
//   },
// };

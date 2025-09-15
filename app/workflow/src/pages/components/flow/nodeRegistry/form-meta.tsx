import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldArray,
} from "@flowgram.ai/free-layout-editor";
import { SideSheet, TextArea, Tooltip, Typography } from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { FormContent } from "../../../../form-components";
import { useIsSidebar } from "../../../../hooks";

import "./index.scss";
import { FormHeader } from "../form-header";
import lodash, { uniqueId } from "lodash";
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { useRef, useState } from "react";
import { IResourceTreeNodeEntity } from "../../../Layout/Design/Resource/entity";
import ResourceEditDrawer from "../../../Layout/Design/Resource/ResourceEditDrawer";

import { PropertyEdit } from "./propertyEdit";
import IoType from "./ioType";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();

  const readonly = !useIsSidebar();

  // const { node } = useNodeRender();

  const { nodeList = [], paramTypes, element: flowElementRef,  runInfo} = useFlowEnviromentContext()

  const {outputResource} = runInfo

  const is_dynamic_input_ = form.getValueIn("is_dynamic_input_");
  const is_dynamic_output_ = form.getValueIn("is_dynamic_output_");

  const key_ = form.getValueIn("key_");

  const name_ = form.getValueIn('name_')

  const ioType = form.getValueIn('io_type_')

  function getIoTypeFieldName() {
    const required_params = form.getValueIn('required_params_')
    if (required_params && required_params.length > 0) {
      return required_params[0]
    }
    return "empty-field"
  }

  function getIoTypeFieldValue() {
    const fieldName = getIoTypeFieldName()
    if (fieldName) {
      return form.getValueIn(fieldName)
    }
    return ""
  }

  function handleIoTypeValueChange(value: string) {
    form.setValueIn(getIoTypeFieldName(), value)
  }





  const excludeFields = [
    "id",
    "key_",
    // "param_",
    // "inputs_",
    // "outputs_",
    // "node_repository_",
    'is_dynamic_input_',
    'is_dynamic_output_',
    "is_graph_",
    "is_inner_",
    "node_type_",

    'developer_',
    'source_',
    'version_',

    'io_type_', 
    'required_params_'

  ];


  const basicFields = lodash.difference(
    Object.keys(form.values),
    excludeFields
  );


  function isTextNode() {
    const textNodes: string[] = ['nndeploy::qwen::PrintNode',]
    if (textNodes.includes(key_)) {
      return true
    }
    return false
  }

  // function needShowTextContent() {

  //   const nodeNames = outputResource.text.map(item => item.name)

  //   const needShow = isTextNode() && nodeNames.includes(form.getValueIn('name_')) //path_.includes('&time=')

  //   return needShow
  // }

  function isInputMediaNode() {
    // const imageNodes: string[] = ['nndeploy::codec::OpenCvImageDecode', 'nndeploy::codec::OpenCvVideoDecode']
    // if (imageNodes.includes(key_)) {
    //   return true
    // }
    // return false

     const nodeType = form.getValueIn('node_type_')
     return nodeType == 'Input'

    // if (ioType) {
    //   return true
    // } else {
    //   return false
    // }
  }

  function isOutputMediaNode() {
    // const imageNodes: string[] = ['nndeploy::codec::OpenCvImageEncode', 'nndeploy::codec::OpenCvVideoEncode']
    // if (imageNodes.includes(key_)) {
    //   return true
    // }
    // return false

     const nodeType = form.getValueIn('node_type_')
     return nodeType == 'Output'
  }


  function isImageFile(filename: string) {

    const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff'];
    const lowerFilename = filename.toLowerCase();
    return imageExtensions.some(ext => lowerFilename.endsWith(ext));
  }

  function isVideoFile(filename: string) {

    const videoExtensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.mpeg', '.mpg'];
    const lowerFilename = filename.toLowerCase();
    return videoExtensions.some(ext => lowerFilename.endsWith(ext));
  }

  const [resourceEdit, setResourceEdit] = useState<IResourceTreeNodeEntity>();
  const [resoureEditVisible, setResoureEditVisible] = useState(false)


  function onShowMediaFile(event: React.MouseEvent<HTMLElement, MouseEvent>, file: string) {
    event.stopPropagation();
    event.preventDefault();

    //const parentId = file.includes('images') ? 'images' : file.includes('videos') ? 'videos' : ''
    //const fileName = file.substring(file.lastIndexOf('/') + 1)
    setResourceEdit({ id: uniqueId(), parentId: "", type: 'leaf', file_info: { saved_path: file } })
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

  function getPopupContainer() {

    let container = renderFormRef?.current;
    while (container && !(container instanceof HTMLElement && container.classList.contains('demo-container'))) {
      container = container.parentElement;
    }
    return container
  }
  // function needShowMedia() {

  //   const nodeNames = outputResource.path.map(item => item.name)
  //   const needShow = isInputMediaNode() || (isOutputMediaNode() && nodeNames.includes(form.getValueIn('name_')))  //path_.includes('&time=')

  //   return needShow
  // }

  return (
    <div className="drawer-render-form" ref={renderFormRef}>
      <FormHeader />

      <FormContent>
        {!isSidebar && (
          <>
            <div className="connection-area">
              <div className="input-area">
                <FieldArray name="inputs_">
                  {({ field }) => {

                    return <>
                      {field.map((child, index) => {
                        return (
                          <Field<any> key={child.name} name={child.name}>
                            {({ field: childField, fieldState: childState }) => {

                              console.log('inputs_ childField.value.id', childField.value.id)
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
                                    </>}>{childField.value.type_}</Tooltip>
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

                            //console.log('outputs_ childField.value.id', childField.value.id)
                            return <>

                              {/* <FormItem
                              name={`${childField.value.type_}`}///${childField.value.desc_}
                              description={<> <p>type: {childField.value.type_}</p>
                                <p>desc: {childField.value.desc_}</p></>}
                              type="boolean"
                              required={false}

                            > */}

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
                                    justifyContent: 'end',
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
                                  </>}>{childField.value.type_}</Tooltip>
                                </div>

                                <div
                                  style={{
                                    flexGrow: 1,
                                    minWidth: 0,
                                  }}
                                >

                                  <div
                                    className="connection-point connection-point-right"
                                    data-port-id={childField.value.id}
                                    data-port-type="output"
                                    data-port-desc={childField.value.desc_}
                                  >

                                  </div>
                                </div>
                              </div>


                              {/* </FormItem> */}
                            </>
                          }}
                        </Field>
                      ))}
                    </>
                  }}
                </FieldArray>
              </div>
            </div>

            {/* <Field key={'path_'} name={'path_'}>
              {({ field, fieldState }) => {

                if (needShowMedia() && field.value && isImageFile(field.value as string)) {

                  const filePath = outputResource.path.find(item => item.name == name_)?.path ?? field.value

                  const url = `/api/preview?file_path=${filePath}` //${field.value}&time=${Date.now()}
                  return <div className="resource-preview zhang">
                    <img src={url} onClick={((event) => onShowMediaFile(event, field.value as string))}

                    />
                  </div>
                } else if (needShowMedia() && field.value && isVideoFile(field.value as string)) {

                  const filePath = outputResource.path.find(item => item.name == name_)?.path ?? field.value

                  const url = `/api/preview?file_path=${filePath}`  //${field.value}&time=${Date.now()}
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

            </Field> */}

            {
              (isInputMediaNode() || isOutputMediaNode()) && getIoTypeFieldName() &&

              <Field key={getIoTypeFieldName()} name={getIoTypeFieldName()} >
                {({ field, fieldState }) => {
                  return <IoType
                    direction = {isInputMediaNode() ? 'input' : 'output'}
                    ioDataType={ioType}

                   // value={getIoTypeFieldValue()}
                    value = {field.value as string}
                    //onChange={handleIoTypeValueChange}
                    onChange = {field.onChange}
                  />
                }}
              </Field>
            }
            
            {/* {
              isTextNode() && needShowTextContent() &&
              <TextArea rows={8} value={outputResource.text.find(item => item.name == form.getValueIn('name_'))?.text}>

              </TextArea>
            } */}

          </>
        )}
        {isSidebar ? (
          <>
            <div className="property-container">
              <div className="UIProperties">

                {basicFields.map((fieldName, index) => {

                  return (
                    <Field key={fieldName} name={fieldName} >
                      {({ field, fieldState }) => {
                        return <PropertyEdit fieldName={fieldName} parentPaths={[]}
                          // value={form.getValueIn(fieldName)}

                          // onChange={(value) => {
                          //   form.setValueIn(fieldName, value)
                          // }}

                          value={field.value}

                          onChange={(value) => {
                            field.onChange(value)
                          }}
                          onRemove={() => { }}
                          onFieldRename={() => {

                          }}

                          showLine={false}

                          form={form}
                          nodeList={nodeList}
                          paramTypes={paramTypes}
                          isLast={index == basicFields.length - 1}
                          topField={true}

                        />
                      }}
                    </Field>

                  )
                })}

              </div>
            </div>

            {/* {is_dynamic_input_ && (
              <Section text={"inputs_"} key="inputs_">
                <FormDynamicPorts portType="inputs_" />
              </Section>
            )}

            {is_dynamic_output_ && (
              <Section text={"outputs_"} key="outputs_">
                <FormDynamicPorts portType="outputs_" />
              </Section>
            )}
            {
              form.values.hasOwnProperty('param_') && <Section text={"param_"}>
                <FormParams />
              </Section>
            }

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
            {
              isTextNode() && needShowTextContent() &&
              <TextArea rows={8} value={outputResource.text.find(item => item.name == form.getValueIn('name_'))?.text}>

              </TextArea>
            } */}

          </>
        ) : (
          <></>
        )}
      </FormContent>


      <SideSheet
        width={"60%"}
        mask={true}
        visible={resoureEditVisible}
        onCancel={handleResoureDrawerClose}
        closeOnEsc={true}
        title={'resource preview'}
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

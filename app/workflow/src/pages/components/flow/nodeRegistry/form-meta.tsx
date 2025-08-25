import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldArray,
} from "@flowgram.ai/free-layout-editor";
import { Button, Popover, Select, SideSheet, Switch, TextArea, Typography, VideoPlayer } from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { Feedback, FormContent } from "../../../../form-components";
import { useIsSidebar } from "../../../../hooks";

import "./index.scss";
import { FormHeader } from "../form-header";
import { FormItem } from "../form-item";
import { FormParams } from "../form-params";
import { FxExpression } from "../../../../form-components/fx-expression";
import Section from "@douyinfe/semi-ui/lib/es/form/section";
import lodash, { uniqueId } from "lodash";
import { FormDynamicPorts } from "../form-dynamic-ports";
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { getFieldType } from "../functions";
import { INodeEntity } from "../../../Node/entity";
import { useRef, useState } from "react";
import NodeRepositoryEditor from "../NodeRepositoryEditor";
import { IResourceTreeNodeEntity } from "../../../Layout/Design/Resource/entity";
import ResourceEditDrawer from "../../../Layout/Design/Resource/ResourceEditDrawer";
import { IconCrossCircleStroked, IconPlus } from "@douyinfe/semi-icons";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();

  const readonly = !useIsSidebar();

  // const { node } = useNodeRender();

  const { nodeList = [], paramTypes, element: flowElementRef, outputResource } = useFlowEnviromentContext()

  const is_dynamic_input_ = form.getValueIn("is_dynamic_input_");
  const is_dynamic_output_ = form.getValueIn("is_dynamic_output_");

  const key_ = form.getValueIn("key_");

  const name_ = form.getValueIn('name_')

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
    "node_type_",
    'developer_',
    'source_',
    'version_',
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

  function needShowTextContent() {

    const nodeNames = outputResource.text.map(item => item.name)

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
  function needShowMedia() {

    const nodeNames = outputResource.path.map(item => item.name)
    const needShow = isInputMediaNode() || (isOutputMediaNode() && nodeNames.includes(form.getValueIn('name_')))  //path_.includes('&time=')

    return needShow
  }


  function isRequiredField(fieldName: string) {

    // var temp = form.getValueIn('param_')
    const required_params = form.getValueIn("required_params_");
    if (required_params && Array.isArray(required_params) && required_params.includes(fieldName)) {
      return true
    }
    return false

  }

  function renderField(fieldName: string, parentPaths: string[]) {
    const fieldType = getFieldType([...parentPaths, fieldName], form, nodeList, paramTypes)

    const filedPath = [...parentPaths, fieldName].join('.')

    var defaulValue = form.getValueIn(filedPath)


    var fieldPath = parentPaths.concat(fieldName).join('.')
    if (fieldType.isArray) {
      return (
        <div className="number-array-field">
          <div className="field-label">{fieldName} {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}</div>
          <div className="filed-array-items">
            <FieldArray name={parentPaths.concat(fieldName).join('.')}>
              {({ field }) => (
                <>
                  {field.map((child, index) => (
                    <Field key={child.name} name={child.name}>
                      {({
                        field: childField,
                        fieldState: childState,
                      }) => (
                        <div className="expression-field" style={{ width: '100%' }}
                        >
                          <>
                            {

                              fieldType.componentType == 'boolean' ?
                                <Switch checked={!!childField.value}
                                  //label='开关(Switch)' 
                                  onChange={(value: boolean) => {
                                    childField.onChange(value)
                                  }} />
                                : fieldType.componentType == 'select' ?
                                  <Select

                                    value={childField.value as number}
                                    style={{ width: '100%' }}
                                    optionList={paramTypes[fieldType.selectKey!].map(item => {
                                      return {
                                        label: item,
                                        value: item
                                      }
                                    })}

                                    onChange={(value) => {
                                      childField.onChange(value)
                                    }}

                                  >

                                  </Select> :

                                  <FxExpression
                                    value={childField.value as number}
                                    fieldType={fieldType}
                                    onChange={(v) => childField.onChange(v)}
                                    icon={
                                      <Button
                                        theme="borderless"
                                        icon={<IconCrossCircleStroked />}
                                        onClick={() => field.delete(index)}
                                      />
                                    }
                                    hasError={
                                      Object.keys(childState?.errors || {})
                                        .length > 0
                                    }
                                    readonly={readonly}
                                  />


                            }
                            <Feedback
                              errors={childState?.errors}
                              invalid={childState?.invalid}
                            />
                          </>
                        </div>
                      )}
                    </Field>
                  ))}
                  {!readonly && (
                    <div>
                      <Button
                        theme="borderless"
                        icon={<IconPlus />}
                        onClick={() => field.append(0)}
                      >
                        Add
                      </Button>
                    </div>
                  )}
                </>
              )}
            </FieldArray>
          </div>
        </div>
      );
    } else if (fieldType.componentType == 'object') { // 如果是对象类型 
      let results: any[] = []
      for (var childFieldName in defaulValue) {
        let result = renderField(childFieldName, [...parentPaths, fieldName])
        results.push(result)
      }

      return <div className="object-field">
        <div className="field-label">
          {fieldName} {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
        </div>
        <div className="child-fields">
          {results}
        </div>

      </div>
    } else {

      return <Field key={fieldPath} name={fieldPath} defaultValue={defaulValue}>
        {({ field, fieldState }) => {

          return <FormItem
            name={fieldName}
            type={"string" as string}
            labelWidth={138}
            required={isRequiredField(fieldName)}
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
                      style={{ width: '100%' }}

                      value={field.value}

                      onChange={(value) => {
                        field.onChange(value)
                      }}
                      optionList={paramTypes[fieldType.selectKey!].map(item => {
                        return {
                          label: item,
                          value: item
                        }
                      })}>

                    </Select> :
                    <FxExpression
                      value={field.value}
                      fieldType={fieldType}
                      onChange={field.onChange}
                      readonly={readonly}
                      hasError={Object.keys(fieldState?.errors || {}).length > 0}
                      icon={<></>}
                    />
              }
            </>
            <Feedback errors={fieldState?.errors} />
          </FormItem>
        }
        }
      </Field>
    }

  }


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
                                <FormItem
                                  name={`${childField.value.type_}`}///${childField.value.desc_}
                                  type="boolean"
                                  description={<> <p>type: {childField.value.type_}</p>
                                    <p>desc: {childField.value.desc_}</p></>}
                                  required={false}
                                //labelWidth={40}
                                >
                                  <Popover content={childField.value.desc_} position="right">
                                    <div
                                      className="connection-point connection-point-left"
                                      data-port-id={childField.value.id}
                                      data-port-type="input"
                                      data-port-desc={childField.value.desc_}
                                    //data-port-wangba={childField.value.desc_}
                                    ></div>
                                  </Popover>
                                </FormItem>
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

                            console.log('outputs_ childField.value.id', childField.value.id)
                            return <FormItem
                              name={`${childField.value.type_}`}///${childField.value.desc_}
                              description={<> <p>type: {childField.value.type_}</p>
                                <p>desc: {childField.value.desc_}</p></>}
                              type="boolean"
                              required={false}

                            >

                              <div
                                className="connection-point connection-point-right"
                                data-port-id={childField.value.id}
                                data-port-type="output"
                                data-port-desc={childField.value.desc_}
                              ></div>

                            </FormItem>
                          }}
                        </Field>
                      ))}
                    </>
                  }}
                </FieldArray>
              </div>
            </div>

            <Field key={'path_'} name={'path_'}>
              {({ field, fieldState }) => {

                if (needShowMedia() && field.value && isImageFile(field.value as string)) {

                  const filePath = outputResource.path.find(item => item.name == name_)?.path ?? field.value

                  const url = `/api/preview?file_path=${filePath}` //${field.value}&time=${Date.now()}
                  return <div className="resource-preview">
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

            </Field>
            {
              isTextNode() && needShowTextContent() &&
              <TextArea rows={8} value={outputResource.text.find(item => item.name == form.getValueIn('name_'))?.text}>

              </TextArea>
            }

          </>
        )}
        {isSidebar ? (
          <>
            {basicFields.map((fieldName) => {
              return (
                renderField(fieldName, [])


              );
            })}

            {is_dynamic_input_ && (
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
            }

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

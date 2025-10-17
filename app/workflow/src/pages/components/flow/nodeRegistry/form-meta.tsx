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
import { useEffect, useRef, useState } from "react";
import { IResourceTreeNodeEntity } from "../../../Layout/Design/Resource/entity";
import ResourceEditDrawer from "../../../Layout/Design/Resource/ResourceEditDrawer";

import { PropertyEdit } from "./propertyEdit";
import IoType from "./ioType";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();

  const readonly = !useIsSidebar();

  const { nodeList = [], paramTypes, element: flowElementRef, runInfo } = useFlowEnviromentContext()


  //const key_ = form.getValueIn("key_");

  const registryKey =  form.getValueIn("key_")

  const name_ = form.getValueIn('name_')

  const ioType = form.getValueIn('io_type_')

  function getIoTypeFieldName() {


    const parents = ['', 'param_', 'param']
    for (const parent of parents) {

      const params = ['io_param_', 'required_params_']

      for (const param of params) {
        let ioPosition = form.getValueIn((parent ? parent + '.' : '') + param)
        if (ioPosition && ioPosition.length > 0) {
          return (parent ? parent + '.' : '') + ioPosition[0]
        }
      }
    }

    return "empty-field"
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

   // 'io_type_',
    'io_params_',
    'dropdown_params',
    'required_params_',
    'size',


  ];


  const basicFields = lodash.difference(
    Object.keys(form.values),
    excludeFields
  );


  function isInputMediaNode() {


    const nodeType = form.getValueIn('node_type_')
    return nodeType == 'Input'

  }

  function isOutputMediaNode() {


    const nodeType = form.getValueIn('node_type_')
    return nodeType == 'Output'
  }



  const [resourceEdit, setResourceEdit] = useState<IResourceTreeNodeEntity>();
  const [resoureEditVisible, setResoureEditVisible] = useState(false)


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
            {
              (isInputMediaNode() || isOutputMediaNode()) && getIoTypeFieldName() &&

              <Field key={name_} name={getIoTypeFieldName()} >
                {({ field, fieldState }) => {
                  return <IoType
                    direction={isInputMediaNode() ? 'input' : 'output'}
                    ioDataType={ioType}
                    nodeName={name_}

                    // value={getIoTypeFieldValue()}
                    value={field.value as string}
                    //onChange={handleIoTypeValueChange}
                    onChange={value => {

                      field.onChange(value)
                      // form.setValueIn(getIoTypeFieldName(), value)
                      //setRefresh({})

                    }
                    }
                  />
                }}
              </Field>
            }

          </>
        )}
        {isSidebar ? (
          <div className="property-container">
            <div className="UIProperties">

              {basicFields.map((fieldName, index) => {

                return (
                  <Field key={fieldName} name={fieldName} >
                    {({ field, fieldState }) => {

                      if (field.name == 'param_') {
                        let j = 0;
                      }
                      return <PropertyEdit 
                      fieldName={fieldName} 
                      fieldNameLabel= {fieldName}
                      parentPaths={[]}
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

                        //form={form}
                        registryKey={registryKey}
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

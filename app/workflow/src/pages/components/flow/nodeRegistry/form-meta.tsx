import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldArray,
  useClientContext,
} from "@flowgram.ai/free-layout-editor";

import { SubCanvasRender } from '@flowgram.ai/free-container-plugin';

import {
  //SideSheet, TextArea, 
  Tooltip, Typography
} from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { FormContent } from "../../../../form-components";
import { useIsSidebar, useNodeRenderContext } from "../../../../hooks";

import "./index.scss";
import { FormHeader } from "../form-header";
import lodash, { uniqueId } from "lodash";
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { useEffect, useRef, useState } from "react";
//import { IResourceTreeNodeEntity } from "../../../Layout/Design/Resource/entity";
//import ResourceEditDrawer from "../../../Layout/Design/Resource/ResourceEditDrawer";

import { PropertyEdit } from "./propertyEdit";
import IoType, { EnumIODataType } from "./ioType";
import { isContainerNode } from "../functions";
import classNames from "classnames";
import { getOutputNodePath } from '../../../../components/tools/webui/functions';
import { getResourceFields, resourceFieldName, setResourceValue } from '../../../../components/resource-files';
import { ResourceFilesInput } from '../../../../components/resource-files-input';

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();

  const { nodeList = [], paramTypes, element: flowElementRef, runInfo } = useFlowEnviromentContext()

  const { expanded, node } = useNodeRenderContext();

  const clientContext = useClientContext()


  //const key_ = form.getValueIn("key_");

  const registryKey = form.getValueIn("key_")

  const name_ = form.getValueIn('name_')

  const ioType: EnumIODataType = form.getValueIn('io_type_')
  const resourceFields = getResourceFields(form.values);

  const isContainer = isContainerNode(node.id, clientContext)

  function getIoTypeFieldName() {


    const parents = ['', 'param_', 'param']
    for (const parent of parents) {

      const params = ['io_param_', 'io_params_', 'required_params_']

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
    "name_",
    "key_",
    // "param_",
    // "inputs_",
    // "outputs_",
    // "node_repository_",
    'is_dynamic_input_',
    'is_dynamic_output_',
    "is_graph_",
    "is_composite_node_",
    "is_loop_",
    "is_inner_",
    "node_type_",

    'developer_',
    'source_',
    'version_',

    'io_type_',
    'io_params_',
    'dropdown_params',
    'required_params_',
    'size',
    'blocks',
    "image_url_",
    "video_url_",
    "audio_url_",
    "model_url_",
    "other_url_",


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



  // const [resourceEdit, setResourceEdit] = useState<IResourceTreeNodeEntity>();
  //  const [resoureEditVisible, setResoureEditVisible] = useState(false)


  // function handleResoureDrawerClose() {
  //   setResoureEditVisible(false)
  // }
  // function onResourceEditDrawerClose() {
  //   setResoureEditVisible(false)
  // }
  // function onResourceEditDrawerSure() {
  //   setResoureEditVisible(false)
  // }

  const renderFormRef = useRef<any>();

  function getPopupContainer() {

    let container = renderFormRef?.current;
    while (container && !(container instanceof HTMLElement && container.classList.contains('demo-container'))) {
      container = container.parentElement;
    }
    return container
  }

  return (



    <div className={classNames({
      "drawer-render-form": true,
      isSidebar
    })} ref={renderFormRef}>

      <FormHeader />
      <FormContent>
        {!isSidebar && (
          <>

            {
              resourceFields.length === 0 && (isInputMediaNode() || isOutputMediaNode()) && getIoTypeFieldName() &&

              <Field key={name_} name={getIoTypeFieldName()} >
                {({ field, fieldState }) => {
                  return <IoType
                    direction={isInputMediaNode() ? 'input' : 'output'}
                    ioDataType={ioType}
                    nodeName={name_}
                    nodePath={getOutputNodePath(node)}
                    fieldPath={getIoTypeFieldName().split('.')}

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
            {resourceFields.map(resource => <div key={JSON.stringify(resource.path)} style={{ marginBottom: 8 }}>
              <div>{resource.path.join(' / ')}</div>
              <Field name={resourceFieldName(resource.path)}>
                {() => <ResourceFilesInput label={`${name_} ${resource.path.join(' / ')}`}
                  value={form.getValueIn(resourceFieldName(resource.path))} directory={resource.directory}
                  onChange={value => { form.values = setResourceValue(form.values, resource.path, value); }} />}
              </Field>
            </div>)}
            {
              isContainer &&
              <SubCanvasRender style={{ visibility: 'hidden' }} offsetY={-50} />

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
                        fieldNameLabel={fieldName}
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



      {/* <SideSheet
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

      </SideSheet> */}
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

import { IconChevronDown, IconChevronRight, IconMinus } from "@douyinfe/semi-icons";
import { getFieldType } from "../functions";
import { usePropertiesEdit } from "./hooks";
import classNames from "classnames";
import { IconButton, Input, InputNumber, Select, Switch } from "@douyinfe/semi-ui";
import { IconAddChildren } from "../../json-schema-editor/styles";
import { useState } from "react";
import lodash from 'lodash'
import { useNodeRender, WorkflowNodePortsData } from "@flowgram.ai/free-layout-editor";
import Section from "@douyinfe/semi-ui/lib/es/form/section";
import NodeRepositoryEditor from "../NodeRepositoryEditor";
import { INodeEntity } from "../../../Node/entity";

export function PropertyEdit(
  props: {
    fieldName: string,
    parentPaths: string[],
    value: any,
    onChange: (value: any) => void,
    onFieldRename: (newName: string) => void,
    onRemove?: () => void;
    showLine: boolean,
    isLast: boolean,
    topField?: boolean

    form: any,
    nodeList: any,
    paramTypes: any
  }) {
  const { fieldName, parentPaths, value, onChange: onChangeProps, onRemove, showLine = false,
    isLast = false,
    topField = false,
    form
    , nodeList,
    paramTypes
  } = props

  if (fieldName == 'param_') {
    let j = 0
  }

  const { node } = useNodeRender();
  const ports = node.getData(WorkflowNodePortsData)

  const fieldType = getFieldType([...parentPaths, fieldName], form, nodeList, paramTypes)

  const { propertyList, isDrilldownObject, onAddProperty, onAddObjectProperty, onRemoveProperty, onEditProperty } =
    usePropertiesEdit(fieldName, parentPaths, value, form, nodeList, paramTypes,

      onChangeProps
    )

  let children: React.ReactNode = <></>

  var fieldPath = parentPaths.concat(fieldName).join('.')

  var showCollapse = fieldType.isArray || fieldType.componentType == 'object'

  const [collapse, setCollapse] = useState(false)



  function isRequiredField(fieldName: string) {


    if(fieldName == 'model_value_'){
      let j = 0;
    }

    let required_params: string[] = []
    if (parentPaths.includes('param_')) {
      required_params = form.getValueIn("param_.required_params_");
    } else {
      required_params = form.getValueIn("required_params_");
    }
    if (required_params && Array.isArray(required_params) && required_params.includes(fieldName)) {
      return true
    }

    return false

  }

  if (fieldName == 'node_repository_') {

    let repositories = value ?? []

    if (repositories && repositories.length > 0) {
      return <Section text={"node_repository_"} className="node_repositories">
        {

          <NodeRepositoryEditor node_repository_={repositories as INodeEntity[]}
            nodeList={nodeList} paramTypes={paramTypes}


            onUpdate={(values) => {
              //console.log(values)
              props.onChange(values)

            }} />
        }
      </Section>
    } else {
      return <></>
    }

  }


  if (fieldType.isArray) {
    children = (
      <>
        <div className="UIPropertyMain">
          <div className="UIRow">
            <div className="UIName">
              <div>
              <Input
                value={fieldName}
                required = {isRequiredField(fieldName)}

                onChange={(value) => {
                  //onEditProperty(_property.key!, _v);
                  props.onFieldRename(value)
                }} />
                </div>
              {/* {fieldName}  */}

              {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
            </div>
            <div className="UIActions">
              <IconButton
                size="small"
                theme="borderless"
                icon={<IconAddChildren />}
                onClick={() => {
                  //onAddProperty(fieldType.originValue)
                  if (fieldName == 'inputs_' || fieldName == 'outputs_') {


                    let property = { id: 'port_' + Math.random().toString(36).substr(2, 9), ...fieldType.originValue[0] }

                    onAddProperty(property)
                    setTimeout(() => {
                      ports.updateDynamicPorts()
                    }, 10)

                  } else {
                    onAddProperty(fieldType.originValue[0])
                  }

                }}
              />
              {!topField &&
                <IconButton
                  size="small"
                  theme="borderless"
                  icon={<IconMinus size="small" />}
                  onClick={onRemove}
                />
              }
            </div>
          </div>
        </div>
        <div className={classNames('UICollapsible', { 'collapsed': collapse })}>
          <div className={classNames('UIProperties', { shrink: true })}>
            {propertyList.map((_property, index) => {

              return <PropertyEdit
                fieldName={index + ""}
                parentPaths={[...parentPaths, fieldName]}
                key={_property.key}
                value={_property.value}
                onChange={(_v) => {
                  onEditProperty(_property.key!, { value: _v });
                }}
                onFieldRename={(newName: string) => {
                  onEditProperty(_property.key!, { name: newName });
                }}
                onRemove={() => {
                  onRemoveProperty(_property.key!);
                }}
                isLast={index === propertyList.length - 1}
                showLine={true}
                form={form}
                nodeList={nodeList}
                paramTypes={paramTypes}
              />
            })}
          </div>
        </div>
      </>
    );
  }
  else if (fieldType.componentType == 'object') { // 如果是对象类型 


    children = <>
      <div className="UIPropertyMain">
        <div className="UIRow">
          <div className="UIName">
            <Input
              value={fieldName}
              required = {isRequiredField(fieldName)}

              onChange={(value) => {
                //onEditProperty(_property.key!, _v);
                props.onFieldRename(value)
              }} />  
              {/* {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>} */}
          </div>
          <div className="UIActions">
            <IconButton
              size="small"
              theme="borderless"
              icon={<IconAddChildren />}
              onClick={() => {
                //onAddProperty();
                //onAddProperty(fieldType.originValue)


                let prefix = ''

                let max_number = 0;
                let temp = value
                const keys = Object.keys(fieldType.originValue);

                let firstKey = keys[0]

                prefix = firstKey.split('_')[0]

                keys.map(name => {
                  let parts = name.split('_')


                  if (parts.length > 1 && lodash.isNumber(new Number(parts[parts.length - 1]))) {
                    let temp = new Number(parts[parts.length - 1]).valueOf()
                    if (max_number < temp) {
                      max_number = temp
                    }
                  }
                })

                let fieldName = prefix + '_' + (max_number + 1)


                const firstValue = fieldType.originValue[firstKey];
                onAddObjectProperty(fieldName, firstValue)

              }}
            />
            {!topField &&
              <IconButton
                size="small"
                theme="borderless"
                icon={<IconMinus size="small" />}
                onClick={onRemove}
              />
            }
          </div>
        </div>
      </div>
      <div className={classNames('UICollapsible', { 'collapsed': collapse })}>
        <div className={classNames('UIProperties', { shrink: true })}>
          {propertyList.map((_property, index) => (
            <PropertyEdit
              fieldName={_property.name}
              parentPaths={[...parentPaths, fieldName]}
              // key={_property.key}
              value={_property.value}
              onChange={(_v) => {
                onEditProperty(_property.key!, { value: _v });
              }}
              onFieldRename={(newName: string) => {
                onEditProperty(_property.key!, { name: newName });
              }}
              onRemove={() => {
                onRemoveProperty(_property.key!);
              }}
              isLast={index === propertyList.length - 1}
              showLine={true}
              form={form}
              nodeList={nodeList}
              paramTypes={paramTypes}
            />
          ))}
        </div>
      </div>
    </>



  } else {

    let component = fieldType.componentType == 'boolean' ?
      <Switch checked={!!value}
        //label='开关(Switch)' 
        onChange={(value: boolean) => {
          onChangeProps(value)
        }} />
      : fieldType.componentType == 'select' ?
        <Select
          style={{ width: '100%' }}

          value={value}

          onChange={(value) => {
            onChangeProps(value)
          }}
          optionList={paramTypes[fieldType.selectKey!].map((item: any) => {
            return {
              label: item,
              value: item
            }
          })}>

        </Select> :
        fieldType.primateType == 'number' ?
          <InputNumber
            value={value}
            onChange={(value) => {
              onChangeProps(value)
            }}
          />
          :
          <Input
            value={value}

            onChange={(value) => {
              onChangeProps(value)
            }}
          //readonly={readonly}
          />


    children = <div className="UIPropertyMain">
      <div className="UIRow">
        <div className="UIName">
          {fieldName} {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
        </div>
        <div className="UIComponent">
          {component}
        </div>
        <div className="UIActions">
          {/* <IconButton
              size="small"
              theme="borderless"
              icon={<IconAddChildren />}
              onClick={() => {
                //onAddProperty();
               onAddProperty(fieldType.originValue)

              }}
            /> */}
          {!topField &&
            <IconButton
              size="small"
              theme="borderless"
              icon={<IconMinus size="small" />}
              onClick={onRemove}
            />
          }

        </div>
      </div>
    </div>
  }




  return <>
    <div className={classNames("UIPropertyLeft", { showLine, isLast })}>
      {
        showCollapse && (
          <div className="UICollapseTrigger" onClick={() => setCollapse((_collapse) => !_collapse)}>
            {collapse ? <IconChevronRight size="small" /> : <IconChevronDown size="small" />}
          </div>
        )
      }
    </div>
    <div className="UIPropertyRight">
      {children}
    </div>
  </>
}

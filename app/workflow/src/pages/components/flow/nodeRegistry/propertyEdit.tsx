import { IconChevronDown, IconChevronRight, IconMinus } from "@douyinfe/semi-icons";
import { getFieldType } from "../functions";
import { usePropertiesEdit } from "./hooks";
import classNames from "classnames";
import { IconButton, Input, InputNumber, Select, Switch } from "@douyinfe/semi-ui";
import { IconAddChildren } from "../../json-schema-editor/styles";
import { ReactNode, useEffect, useState } from "react";
import lodash from 'lodash'
import { useNodeRender, WorkflowNodePortsData } from "@flowgram.ai/free-layout-editor";
import Section from "@douyinfe/semi-ui/lib/es/form/section";
import NodeRepositoryEditor from "../NodeRepositoryEditor";
import { INodeEntity } from "../../../Node/entity";

export function PropertyEdit(
  props: {
    fieldName: string,
    fieldNameLabel: ReactNode,
    parentPaths: string[],
    value: any,
    isArrayItem?: boolean,

    onChange: (value: any) => void,
    onFieldRename: (newName: string) => void,
    onRemove?: () => void;
    showLine: boolean,
    isLast: boolean,
    topField?: boolean

    //form: any,
    registryKey: string,
    nodeList: any,
    paramTypes: any
  }) {
  let {
    fieldName, fieldNameLabel, parentPaths, value,
    isArrayItem = false,
    onChange: onChangeProps, onRemove, showLine = false,
    isLast = false,
    topField = false,
    registryKey,
    //form
    nodeList,
    paramTypes
  } = props




  //isArrayItem = false

  if (fieldName == 'user_content_') {
    let j = 0
  }

  function isHiddenField(fieldName: string) {


    const hiddenFields = ['io_params_', 'dropdown_params_', 'required_params_',
      // 'ui_params_'
    ]
    if (hiddenFields.includes(fieldName)) {

      return true
    }

    let is_dynamic_input_ = getFieldType(['is_dynamic_input_'], registryKey, nodeList, paramTypes)?.originValue;

    if (fieldName == 'inputs_') {
      return !is_dynamic_input_
    }



    let is_dynamic_output_ = getFieldType(['is_dynamic_output_'], registryKey, nodeList, paramTypes)?.originValue;
    if (fieldName == 'outputs_') {
      return !is_dynamic_output_
    }

    if (parentPaths.includes('inputs_') || parentPaths.includes('outputs_')) {
      if (fieldName == 'id') {
        return true
      }
    }


    return false
  }

  function isRequiredField(fieldName: string) {


    if (fieldName == 'model_value_') {
      let j = 0;
    }

    let required_params: string[] = []
    if (parentPaths.includes('param_')) {
      required_params = getFieldType(['param_', 'required_params_'], registryKey, nodeList, paramTypes)?.originValue;
    } else {
      required_params = getFieldType(['required_params_'], registryKey, nodeList, paramTypes)?.originValue;
    }
    if (required_params && Array.isArray(required_params) && required_params.includes(fieldName)) {
      return true
    }






    return false

  }


  const fieldType = getFieldType([...parentPaths, fieldName], registryKey, nodeList, paramTypes)

  const { propertyList, isDrilldownObject, onAddProperty, onAddObjectProperty, onRemoveProperty, onEditProperty } =
    usePropertiesEdit(fieldName, parentPaths, value, registryKey, nodeList, paramTypes,

      onChangeProps
    )

  let children: React.ReactNode = <></>

  var fieldPath = parentPaths.concat(fieldName).join('.')

  var showCollapse = fieldType.isArray || fieldType.componentType == 'object'

  const [collapse, setCollapse] = useState(false)
  const [showPortsUpdate, setShowPortsUpdate] = useState(false)
  const [portsUpdateRefresh, setPortsUpdateRefresh] = useState({})

  const [selectOptions, setSelectOptions] = useState<any[]>(fieldType.selectOptions || [])



  const isHidden = isHiddenField(fieldName)

  // if (isHiddenField(fieldName)) {
  //   return <></>
  // }

  const isTopField = topField || parentPaths[parentPaths.length - 1] == 'param_'


  if (fieldName == 'node_repository_') {

    let repositories = value ?? []

    if (repositories && repositories.length > 0) {
      return <Section text={"node_repository_"} className="node_repositories">
        {

          <NodeRepositoryEditor node_repository_={repositories as INodeEntity[]}
            nodeList={nodeList} paramTypes={paramTypes}

            registryKey = {registryKey}
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
            {
              isArrayItem ? <></> :
                <div className="UIName array">

                  {

                    isTopField ? fieldNameLabel : <Input
                      value={fieldName}
                      required={isRequiredField(fieldName)}

                      onChange={(value) => {
                        //onEditProperty(_property.key!, _v);
                        props.onFieldRename(value)
                      }} />
                  }

                  {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
                </div>
            }

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

                    setPortsUpdateRefresh({})
                    setShowPortsUpdate(true)


                  } else {
                    onAddProperty(fieldType.originValue[0])
                  }

                }}
              />
              {
                isTopField ?
                  <></> :
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
                fieldNameLabel={index + ""}
                parentPaths={[...parentPaths, fieldName]}
                key={_property.key}
                value={value[index]} //_property.value
                isArrayItem={_property.isArrayItem}

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
                registryKey={registryKey}
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
          {isArrayItem ? <></> :
            <div className="UIName object">
              {

                isTopField ? fieldNameLabel : <Input
                  value={fieldName}
                  required={isRequiredField(fieldName)}

                  onChange={(value) => {
                    //onEditProperty(_property.key!, _v);
                    props.onFieldRename(value)
                  }} />
              }
              {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
            </div>
          }
          {
            fieldName == 'param_' || isTopField ? <></> :
              <div className="UIActions">
                {
                  fieldName == 'param_' ? <></> :

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
                }
                {isTopField ? <></> :
                  <IconButton
                    size="small"
                    theme="borderless"
                    icon={<IconMinus size="small" />}
                    onClick={onRemove}
                  />
                }





              </div>
          }

        </div>
      </div >
      <div className={classNames('UICollapsible', { 'collapsed': collapse })}>
        <div className={classNames('UIProperties', { shrink: true })}>
          {propertyList.map((_property, index) => {

            if (_property.name.includes('user_content_')) {
              let j = 0;
            }
            return <PropertyEdit
              fieldName={_property.name}
              fieldNameLabel={_property.name}
              parentPaths={[...parentPaths, fieldName]}
              // key={_property.key}
              value={value[_property.name]} //_property.value
              isArrayItem={_property.isArrayItem}

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
              registryKey={registryKey}
              nodeList={nodeList}
              paramTypes={paramTypes}
            />
          }
          )}
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
          filter={true}
          showClear={true}

          onChange={(value) => {
            // 当用户输入新值时自动添加选项
            // if (!fieldType.selectOptions?.some(opt => opt === value)) {
            //   setSelectOptions(prev => [...prev, { label: value, value }]);
            // }
            onChangeProps(value)
          }}

          onSearch={(value) => {

            if (value) {

              if (!fieldType.selectOptions?.some(opt => opt === value)) {
                setSelectOptions(prev => [...(fieldType.selectOptions ?? []), value]);
              }
            } else {
              setSelectOptions(fieldType.selectOptions ?? [])
            }
          }}

          optionList={selectOptions?.map((item: any) => {
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
        {
          isArrayItem ? <></> :
            <div className="UIName">

              {isTopField ? fieldNameLabel : <Input
                value={fieldName}
                required={isRequiredField(fieldName)}

                onChange={(value) => {
                  //onEditProperty(_property.key!, _v);
                  props.onFieldRename(value)
                }} />
              }
              {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}
            </div>
        }
        <div className="UIComponent">
          {component}
        </div>
        {
          isTopField ? <></> :
            <div className="UIActions">

              <IconButton
                size="small"
                theme="borderless"
                icon={<IconMinus size="small" />}
                onClick={onRemove}
              />



            </div>
        }
      </div>
    </div>
  }




  return <>
    <div className={classNames("UIPropertyLeft", { showLine, isLast, isHidden })}>
      {
        showCollapse && (
          <div className="UICollapseTrigger" onClick={() => setCollapse((_collapse) => !_collapse)}>
            {collapse ? <IconChevronRight size="small" /> : <IconChevronDown size="small" />}
          </div>
        )
      }
    </div>
    <div className={classNames("UIPropertyRight", { isHidden })}>

      {children}
    </div>
    {
      showPortsUpdate ? <PortsUpdate depends={portsUpdateRefresh} /> : <></>

    }

  </>
}

interface IPortsUpdateProps {
  depends: any
}

const PortsUpdate: React.FC<IPortsUpdateProps> = (props) => {


  const { node } = useNodeRender();
  const ports = node.getData(WorkflowNodePortsData)

  useEffect(() => {
    setTimeout(() => {
      ports.updateDynamicPorts()
    })

  }, [props.depends])

  return <></>
}


import { Field, FieldArray, useForm } from "@flowgram.ai/free-layout-editor";

import { FormItem } from "../form-item";
import { useIsSidebar } from "../../../../hooks";
import { JsonSchema } from "../../../../typings";
import { FxExpression } from "../../../../form-components/fx-expression";
import { Feedback } from "../../../../form-components";
import { Button, Select, Switch } from "@douyinfe/semi-ui";
import { IconCrossCircleStroked, IconPlus } from "@douyinfe/semi-icons";
import lodash from 'lodash'
import './index.scss'
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { getFieldType } from "../functions";

export function FormParams() {
  const readonly = !useIsSidebar();

  const { nodeList = [], paramTypes } = useFlowEnviromentContext()

  const form = useForm()
   const registryKey = form.values['key_']

  //console.log('form.values', form.values);


  if (readonly) {
    return <></>;
  }

  function isRequiredField(fieldName: string) {

    // var temp = form.getValueIn('param_')
    const required_params = form.getValueIn("param_.required_params_");
    if (required_params && Array.isArray(required_params) && required_params.includes(fieldName)) {
      return true
    }
    return false

  }
  function renderField(fieldName: string, parentPaths: string[]) {


    var fieldPath = parentPaths.concat(fieldName).join('.')

    if (fieldPath == 'param_.input_shape_.1') {
      let k = 0
      //fieldPath = 'param_.input_shape_[0]'
    }

    const fieldType = getFieldType([...parentPaths, fieldName], registryKey, nodeList, paramTypes)






    var defaulValue = form.getValueIn([...parentPaths, fieldName].join('.'))
    if (fieldType.isArray) {
      return (
        <div className="number-array-field">
          <div className="field-label">{fieldName} {isRequiredField(fieldName) ? <span style={{ color: 'rgb(249, 57, 32)' }}>*</span> : <></>}</div>
          <div className="filed-array-items">
            <FieldArray name={fieldPath}>
              {({ field, fieldState }) => {

                if (fieldPath == 'param_.input_shape_.0') {
                  let j = 0
                  let tmp = fieldState
                  let k = 0
                }


                try {
                  field.map((child, index) => {

                    var temp = child
                    var i = 0
                  })
                } catch (e) {

                  return (field.value as any).map((child, index) => {

                    return renderField(index, [...parentPaths, fieldName])
                  })
                }


                return <>
                  {field.map((child, index) => {
                    var childPath = child.name.split('.')
                    if (child.name.includes('param_.input_shape_.0')) {
                      let j = 0
                    }
                    return renderField(childPath[childPath.length - 1], [...parentPaths, fieldName])
              
                  })}
                  {!readonly && (
                    <div>
                      <Button
                        theme="borderless"
                        icon={<IconPlus />}
                        onClick={() => {
                          let lastIsArray = lodash.isArray(field.value[field.value.length - 1])
                          if (lastIsArray) {

                            let temp = [...field.value[field.value.length - 1]]
                            field.append([...temp])
                          } else {
                            let temp = { ...field.value[field.value.length - 1] }
                            field.append({ ...temp })
                          }

                          setTimeout(() => {
                            console.log('form.values', form.values)
                          })

                        }
                        }
                      >
                        Add
                      </Button>
                    </div>
                  )}
                </>
              }}
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
    <Field<any> name="param_">
      {({ field: params }) => {
        const properties = params.value;
        if (!properties) {
          return <></>;
        }
        const content = Object.keys(properties).map((key) => {
          const property = properties[key];
          if (key == 'required_params_') {
            return <></>
          }

          return renderField(key, ['param_'])

        });
        return <>{content}</>;
      }}
    </Field>
  );
}

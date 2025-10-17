import { Field, FieldArray, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { Button } from "@douyinfe/semi-ui";
import { IconPlus, IconCrossCircleStroked } from "@douyinfe/semi-icons";

import { useIsSidebar } from "../../../../hooks";
import { FormItem } from "../form-item";
import { FxExpression } from "../../../../form-components/fx-expression";
import { Feedback } from "../../../../form-components";
import "./styles.scss";
import { random } from "lodash";

interface FormDynamicPortsProps {
  portType: "inputs_" | "outputs_";
}

export const FormDynamicPorts: React.FC<FormDynamicPortsProps> = (props) => {
  const { portType } = props;

  const readonly = !useIsSidebar();
  const { form } = useNodeRender()

  return (
    <FieldArray name={portType}>
      {({ field }) => (
        <>
          {field.map((child, index) => (
            <Field<any> key={child.name} name={child.name}>
              {({ field: childField, fieldState: childState }) => (
                <div className="dynamic-collecion-row">
                  <div className="collection-type">
                    <FormItem
                      name="type_"
                      type="string"
                      required={true}
                      labelWidth={40}
                    >
                      <FxExpression
                        value={childField.value.type_}
                        fieldType={{ isArray: false, primateType: 'string' }}
                        onChange={(v) => {
                          childField.onChange({
                            ...childField.value,
                            type_: v,
                          });
                        }}
                        icon={<></>}
                        hasError={
                          Object.keys(childState?.errors || {}).length > 0
                        }
                        readonly={readonly}
                      />
                      <Feedback
                        errors={childState?.errors}
                        invalid={childState?.invalid}
                      />
                    </FormItem>
                  </div>
                  <div className="collection-desc">
                    <FormItem
                      name="desc_"
                      type="string"
                      required={true}
                      labelWidth={40}
                    >
                      <FxExpression
                        value={childField.value.desc_}
                        fieldType={{ isArray: false, primateType: 'string' }}
                        onChange={(v) => {
                          childField.onChange({
                            ...childField.value,
                            desc_: v,
                          });
                        }}
                        icon={<></>}
                        hasError={
                          Object.keys(childState?.errors || {}).length > 0
                        }
                        readonly={readonly}
                      />
                      <Feedback
                        errors={childState?.errors}
                        invalid={childState?.invalid}
                      />
                    </FormItem>
                  </div>
                  <Button
                    theme="borderless"
                    icon={<IconCrossCircleStroked />}
                    onClick={() => field.delete(index)}
                  />
                </div>
              )}
            </Field>
          ))}
          {!readonly && (
            <div>
              <Button
                theme="borderless"
                icon={<IconPlus />}
                onClick={() => {

                  // function update() {
                  //   form?.setValueIn('all', {})
                  // }
                  // update()
                  var temp = form?.getValueIn(portType)

                  // setTimeout(() => {
                  //   form?.setValueIn(portType, [...form?.getValueIn(portType), {
                  //     id: 'new' + Math.random().toString(36).substr(2, 9),
                  //     ///@ts-ignore
                  //     type_: field.value[field.value.length - 1]?.type_ || '',
                  //     ///@ts-ignore
                  //     desc_: field.value[field.value.length - 1]?.desc_ +  Math.random().toString(36).substr(2, 9)|| ''
                  //   }])
                  // }, 100)

                  const newFields = new Array(5).fill(true).map(() => {
                    return {
                      id:  Math.random().toString(36).substr(2, 9),
                      ///@ts-ignore
                      type_: field.value[field.value.length - 1]?.type_ || '',
                      ///@ts-ignore
                      desc_: field.value[field.value.length - 1]?.desc_ || '',
                    }
                  })

                  var length = form?.getValueIn(portType).length

                   form?.setValueIn(portType, [...form?.getValueIn(portType), 
                  ...newFields])

                  setTimeout(()=>{
                       form?.setValueIn(portType, [...form?.getValueIn(portType).slice(0, length +1), 
                      ])
                  }, 0)

                  // field.append(new Array(5).fill(ture){
                  //   id: 'new' + Math.random().toString(36).substr(2, 9),
                  //   ///@ts-ignore
                  //   type_: field.value[field.value.length - 1]?.type_ || '',
                  //   ///@ts-ignore
                  //   desc_: field.value[field.value.length - 1]?.desc_ || '',
                  // })
                }
                }
              >
                Add
              </Button>
            </div>
          )}
        </>
      )}
    </FieldArray>
  );
};

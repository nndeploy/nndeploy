import { Field, FieldArray, useForm } from "@flowgram.ai/free-layout-editor";

import { FormItem } from "../form-item";
import { useIsSidebar } from "../../../../hooks";
import { JsonSchema } from "../../../../typings";
import { FxExpression } from "../../../../form-components/fx-expression";
import { Feedback } from "../../../../form-components";
import { Button } from "@douyinfe/semi-ui";
import { IconCrossCircleStroked, IconPlus } from "@douyinfe/semi-icons";
import lodash from 'lodash'
import './index.scss'
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { getFieldType } from "../functions";

export function FormParams() {
  const readonly = !useIsSidebar();

  const { nodeList = []}  = useFlowEnviromentContext()

   const form =  useForm()

   console.log('form.values', form.values);


  if (readonly) {
    return <></>;
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

          const fieldType = getFieldType(['param_', key], form, nodeList)

          if (fieldType.isArray) {
            return (
              <div className="number-array-field">
                <div className="field-label">{key}</div>
                <div className="filed-array-items">
                  <FieldArray name={`param_.${key}`}>
                    {({ field }) => (
                      <>
                        {field.map((child, index) => (
                          <Field key={child.name} name={child.name}>
                            {({
                              field: childField,
                              fieldState: childState,
                            }) => (
                              <div className="expression-field"
                              >
                                <FxExpression
                                  value={childField.value as number}
                                  fieldType = {fieldType}
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
                                <Feedback
                                  errors={childState?.errors}
                                  invalid={childState?.invalid}
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
          }
          return (
            <Field key={key} name={`param_.${key}`} defaultValue={property}>
              {({ field, fieldState }) => {

              

                return <FormItem
                  name={key}
                  type={"string" as string}
                  //required={required.includes(key)}
                >
                  <FxExpression
                    value={field.value}
                    fieldType = {fieldType}
                    onChange={field.onChange}
                    readonly={readonly}
                    hasError={Object.keys(fieldState?.errors || {}).length > 0}
                    icon={<></>}
                  />
                  <Feedback errors={fieldState?.errors} />
                </FormItem>
                }
              }
            </Field>
          );
        });
        return <>{content}</>;
      }}
    </Field>
  );
}

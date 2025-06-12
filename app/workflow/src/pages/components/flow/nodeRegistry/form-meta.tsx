import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldRenderProps,
  FieldArray,
} from "@flowgram.ai/free-layout-editor";
import { Typography, Button } from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { Feedback, FormContent } from "../../../../form-components";
import { ConditionInputs } from "../../../../nodes/condition/condition-inputs";
import { useIsSidebar } from "../../../../hooks";

import { ConditionPort } from "./styles";
import "./index.scss";
import { FormHeader } from "../form-header";
import { FormItem } from "../form-item";
import { FormParams } from "../form-params";
import { FxExpression } from "../../../../form-components/fx-expression";
import Section from "@douyinfe/semi-ui/lib/es/form/section";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const readonly = !useIsSidebar();
  console.log('form.values', form.values) 

   const basicFields = ['name_', "device_type_", "type_"].filter(item=>form.values.hasOwnProperty(item))

  return (
    <>
      <FormHeader />
      <FormContent>
        {readonly && (
          <div className="connection-area">
            <div className="input-area">
              <FieldArray name="inputs_">
                {({ field }) => (
                  <>
                    {field.map((child, index) => (
                      <Field<any> key={child.name} name={child.name}>
                        {({ field: childField, fieldState: childState }) => (
                          <FormItem
                            name={`${childField.value.type_}/${childField.value.name_}`}
                            type="boolean"
                            required={false}
                            //labelWidth={40}
                          >
                            <div
                              className="connection-point connection-point-left"
                              data-port-id={childField.value.name_}
                              data-port-type="input"
                            ></div>
                          </FormItem>
                        )}
                      </Field>
                    ))}
                  </>
                )}
              </FieldArray>
            </div>
            <div className="output-area">
              <FieldArray name="outputs_">
                {({ field }) => (
                  <>
                    {field.map((child, index) => (
                      <Field<any> key={child.name} name={child.name}>
                        {({ field: childField, fieldState: childState }) => (
                          <FormItem
                            name={`${childField.value.type_}/${childField.value.name_}`}
                            type="boolean"
                            required={false}
                            //labelWidth={40}
                          >
                            <div
                              className="connection-point connection-point-right"
                              data-port-id={childField.value.name_}
                              data-port-type="output"
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
        )}
     
        <Section text={"basic"}>
          {
            basicFields.map(fieldName=>{
            return  <Field key={fieldName} name={fieldName}>
            {({ field, fieldState }) => (
              <FormItem
                name={fieldName}
                type={"string" as string}
                required={true}
              >
                <FxExpression
                  value={field.value as string}
                  onChange={field.onChange}
                  readonly={readonly}
                  hasError={Object.keys(fieldState?.errors || {}).length > 0}
                  icon={<></>}
                />
                <Feedback errors={fieldState?.errors} />
              </FormItem>
            )}
          </Field>
            })
          }
         
        </Section>
        <Section text={"param_"}>
          <FormParams />
        </Section>
      </FormContent>
    </>
  );
};

export const formMeta: FormMeta<FlowNodeJSON> = {
  render: renderForm,
  validateTrigger: ValidateTrigger.onChange,
  validate: {
    title: ({ value }: { value: string }) =>
      value ? undefined : "Title is required",
    "inputsValues.conditions.*": ({ value }) => {
      if (!value?.value?.content) return "Condition is required";
      return undefined;
    },
  },
};

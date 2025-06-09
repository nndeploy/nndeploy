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
import { FormContent } from "../../../../form-components";
import { ConditionInputs } from "../../../../nodes/condition/condition-inputs";
import { useIsSidebar } from "../../../../hooks";

import { ConditionPort } from "./styles";
import './index.scss'
import { FormHeader } from "../form-header";
import { FormItem } from "../form-item";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const readonly = !useIsSidebar();
  return (
    <>
      <FormHeader />
      <FormContent>
        {
          readonly &&  <div className="connection-area">
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
                          <div className="connection-point connection-point-left"
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
                          <div className="connection-point connection-point-right"
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
        }
       
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

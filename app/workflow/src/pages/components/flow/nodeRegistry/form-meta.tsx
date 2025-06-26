import {
  FormRenderProps,
  FormMeta,
  ValidateTrigger,
  Field,
  FieldArray,
  useFieldValidate,
  useWatch,
  useWatchFormValueIn,
  useNodeRender,
  useWatchFormValues,
} from "@flowgram.ai/free-layout-editor";
import { Typography } from "@douyinfe/semi-ui";

import { FlowNodeJSON } from "../../../../typings";
import { Feedback, FormContent } from "../../../../form-components";
import { useIsSidebar } from "../../../../hooks";

import "./index.scss";
import { FormHeader } from "../form-header";
import { FormItem } from "../form-item";
import { FormParams } from "../form-params";
import { FxExpression } from "../../../../form-components/fx-expression";
import Section from "@douyinfe/semi-ui/lib/es/form/section";
import { GroupNodeRender } from "../../../../components";
import lodash, { random } from "lodash";
import { FormDynamicPorts } from "../form-dynamic-ports";

const { Text } = Typography;

export const renderForm = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const readonly = !useIsSidebar();

  const { node } = useNodeRender();

  //console.log("form.values", form.values);

  // const basicFields = ["name_", "device_type_", "type_"].filter((item) =>
  //   form.values.hasOwnProperty(item)
  // );

  const is_dynamic_input_ = form.getValueIn("is_dynamic_input_");
  const is_dynamic_output_ = form.getValueIn("is_dynamic_output_");

  const excludeFields = [
    "key_",
    "param_",
    "inputs_",
    "outputs_",
    "node_repository_",
    'is_dynamic_input_', 
    'is_dynamic_output_'
  ];
  const basicFields = lodash.difference(
    Object.keys(form.values),
    excludeFields
  );

  //var inputValues = useWatch("inputs_")
  // const whole = useWatchFormValues(node )
  // const  inputValues = whole["inputs_"]

  return (
    <div className="drawer-render-form">
      <FormHeader />

      <FormContent>
        {readonly && (
          <div className="connection-area">
            <div className="input-area">
              <FieldArray name="inputs_">
                {({ field }) => (
                  <>
                    {field.map((child, index) => {
                      return (
                        <Field<any> key={child.name} name={child.name}>
                          {({ field: childField, fieldState: childState }) => {
                            return (
                              <FormItem
                                name={`${childField.value.type_}`}///${childField.value.desc_}
                                type="boolean"
                                required={false}
                                //labelWidth={40}
                              >
                                <div
                                  className="connection-point connection-point-left"
                                  data-port-id={childField.value.id}
                                  data-port-type="input"
                                  data-port-desc={childField.value.desc_}
                                  //data-port-wangba={childField.value.desc_}
                                ></div>
                              </FormItem>
                            );
                          }}
                        </Field>
                      );
                    })}
                  </>
                )}
              </FieldArray>
              {/* {
              inputValues.map((item) => {
                
                return (
                 
                  <FormItem
                   key={`${item.type_}/${item.desc_}`}
                    name={`${item.type_}/${item.desc_}`}
                    type="boolean"
                    required={false}
                    //labelWidth={40}
                  >
                    <div
                      className="connection-point connection-point-left"
                      data-port-id={item.desc_}
                      data-port-type="input"
                    ></div>
                  </FormItem>
                )
              }
              )
            } */}
            </div>
            <div className="output-area">
              <FieldArray name="outputs_">
                {({ field }) => (
                  <>
                    {field.map((child, index) => (
                      <Field<any> key={child.name} name={child.name}>
                        {({ field: childField, fieldState: childState }) => (
                          <FormItem
                            name={`${childField.value.type_}`}///${childField.value.desc_}
                            type="boolean"
                            required={false}
                            //labelWidth={40}
                          >
                            <div
                              className="connection-point connection-point-right"
                              data-port-id={childField.value.id}
                              data-port-type="output"
                              data-port-desc={childField.value.desc_}
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
        {!readonly ? (
          <>
            {/* <Section text={"basic"}> */}
              {basicFields.map((fieldName) => {
                return (
                  <Field key={fieldName} name={fieldName}>
                    {({ field, fieldState }) => (
                      <FormItem
                        name={fieldName}
                        type={"string" as string}
                        required={true}
                      >
                        <FxExpression
                          value={field.value as string}
                          fieldType={{isArray: false, primateType: 'string'}}
                          onChange={field.onChange}
                          readonly={readonly}
                          hasError={
                            Object.keys(fieldState?.errors || {}).length > 0
                          }
                          icon={<></>}
                        />
                        <Feedback errors={fieldState?.errors} />
                      </FormItem>
                    )}
                  </Field>
                );
              })}
            {/* </Section> */}

            {is_dynamic_input_ && (
              <Section text={"inputs_"}>
                <FormDynamicPorts portType="inputs_" />
              </Section>
            )}

            {is_dynamic_output_ && (
              <Section text={"outputs_"}>
                <FormDynamicPorts portType="outputs_" />
              </Section>
            )}

            <Section text={"param_"}>
              <FormParams />
            </Section>
          </>
        ) : (
          <></>
        )}
      </FormContent>
    </div>
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

// export const groupFormMeta: FormMeta<FlowNodeJSON> = {
//   render: GroupNodeRender,
//   validateTrigger: ValidateTrigger.onChange,
//   validate: {
//     title: ({ value }: { value: string }) =>
//       value ? undefined : "Title is required",
//     "inputsValues.conditions.*": ({ value }) => {
//       if (!value?.value?.content) return "Condition is required";
//       return undefined;
//     },
//   },
// };

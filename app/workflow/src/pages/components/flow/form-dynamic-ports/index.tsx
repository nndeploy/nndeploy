import { Field, FieldArray } from "@flowgram.ai/free-layout-editor";
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
                onClick={() =>
                  field.append({
                    id: Math.random().toString(36).substr(2, 9), 
                    type_: "",
                    desc_: '',
                  })
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

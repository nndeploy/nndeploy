import { Field } from "@flowgram.ai/free-layout-editor";

import { FormItem } from "../form-item";
import { useIsSidebar } from "../../../../hooks";
import { JsonSchema } from "../../../../typings";
import { FxExpression } from "../../../../form-components/fx-expression";
import { Feedback } from "../../../../form-components";

export function FormParams() {
  const readonly = !useIsSidebar();

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
          return (
            <Field key={key} name={`param_.${key}`} defaultValue={property}>
              {({ field, fieldState }) => (
                <FormItem
                  name={key}
                  type={"string" as string}
                  //required={required.includes(key)}
                >
                  <FxExpression
                    value={field.value}
                    onChange={field.onChange}
                    readonly={readonly}
                    hasError={Object.keys(fieldState?.errors || {}).length > 0}
                    icon={<></>}
                  />
                  <Feedback errors={fieldState?.errors} />
                </FormItem>
              )}
            </Field>
          );
        });
        return <>{content}</>;
      }}
    </Field>
  );
}

import { Field } from '@flowgram.ai/free-layout-editor';

import { FxExpression } from '../fx-expression';
import { FormItem } from '../form-item';
import { Feedback } from '../feedback';
import { JsonSchema } from '../../typings';
import { useIsSidebar } from '../../hooks';

export function FormInputs() {
  const readonly = !useIsSidebar();
  return (
    <Field<JsonSchema> name="inputs">
      {({ field: inputsField }) => {
        const required = inputsField.value?.required || [];
        const properties = inputsField.value?.properties;
        if (!properties) {
          return <></>;
        }
        const content = Object.keys(properties).map((key) => {
          const property = properties[key];
          return (
            <Field key={key} name={`inputsValues.${key}`} defaultValue={property.default}>
              {({ field, fieldState }) => (
                <FormItem
                  name={key}
                  type={property.type as string}
                  required={required.includes(key)}
                >
                  <FxExpression
                    value={field.value}
                    onChange={field.onChange}
                    readonly={readonly}
                    hasError={Object.keys(fieldState?.errors || {}).length > 0}
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

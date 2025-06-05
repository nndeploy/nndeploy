import { useState } from "react";
import { JsonSchemaEditor } from "../components/json-schema-editor";
import { JsonSchema } from "../components/type-selector/types";

export default function Home() {
  const [value, setValue] = useState<JsonSchema>();
  return (
    <div>
      <h2>Home</h2>
      <JsonSchemaEditor
        value={value}
        onChange={(value: JsonSchema) => {
          var i = 0;
          setValue(value);
        }}
      />
    </div>
  );
}

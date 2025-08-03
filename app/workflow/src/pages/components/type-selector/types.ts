export type BasicType = 'boolean' | 'string' | 'integer' | 'number' | 'object' | 'array';

export interface JsonSchema<T = string> {
  type?: T;
  default?: any;
  title?: string;
  description?: string;
  enum?: (string | number)[];
  properties?: Record<string, JsonSchema>;
  additionalProperties?: JsonSchema;
  items?: JsonSchema;
  required?: string[];
  $ref?: string;
  extra?: {
    order?: number;
    literal?: boolean; // is literal type
    formComponent?: string; // Set the render component
  };
}

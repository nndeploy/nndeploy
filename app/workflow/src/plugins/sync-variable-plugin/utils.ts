import { get } from 'lodash-es';
import { ASTFactory, type ASTNodeJSON } from '@flowgram.ai/free-layout-editor';

import { type JsonSchema } from '../../typings';

/**
 * Sorts the properties of a JSON schema based on the 'extra.index' field.
 * If the 'extra.index' field is not present, the property will be treated as having an index of 0.
 *
 * @param properties - The properties of the JSON schema to sort.
 * @returns A sorted array of property entries.
 */
function sortProperties(properties: Record<string, JsonSchema>) {
  return Object.entries(properties).sort(
    (a, b) => (get(a?.[1], 'extra.index') || 0) - (get(b?.[1], 'extra.index') || 0)
  );
}

/**
 * Converts a JSON schema to an Abstract Syntax Tree (AST) representation.
 * This function recursively processes the JSON schema and creates corresponding AST nodes.
 *
 * For more information on JSON Schema, refer to the official documentation:
 * https://json-schema.org/
 *
 * Note: Depending on your business needs, you can use your own Domain-Specific Language (DSL)
 * Create a new function to convert your custom DSL to AST directly.
 *
 * @param jsonSchema - The JSON schema to convert.
 * @returns An AST node representing the JSON schema, or undefined if the schema type is not recognized.
 */
export function createASTFromJSONSchema(jsonSchema: JsonSchema): ASTNodeJSON | undefined {
  const { type } = jsonSchema || {};

  if (!type) {
    return undefined;
  }

  switch (type) {
    case 'object':
      return ASTFactory.createObject({
        properties: sortProperties(jsonSchema.properties || {}).map(([key, _property]) => ({
          key,
          type: createASTFromJSONSchema(_property),
          meta: { description: _property.description },
        })),
      });
    case 'array':
      return ASTFactory.createArray({
        items: createASTFromJSONSchema(jsonSchema.items!),
      });
    case 'string':
      return ASTFactory.createString();
    case 'number':
      return ASTFactory.createNumber();
    case 'boolean':
      return ASTFactory.createBoolean();
    case 'integer':
      return ASTFactory.createInteger();

    default:
      // If the type is not recognized, return CustomType
      return ASTFactory.createCustomType({ typeName: type });
  }
}

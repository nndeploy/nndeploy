import {
  definePluginCreator,
  FlowNodeVariableData,
  getNodeForm,
  PluginCreator,
  FreeLayoutPluginContext,
  ASTFactory,
} from '@flowgram.ai/free-layout-editor';

import { createASTFromJSONSchema } from './utils';

export interface SyncVariablePluginOptions {}

/**
 * Creates a plugin to synchronize output data to the variable engine when nodes are created or updated.
 * @param ctx - The plugin context, containing the document and other relevant information.
 * @param options - Plugin options, currently an empty object.
 */
export const createSyncVariablePlugin: PluginCreator<SyncVariablePluginOptions> =
  definePluginCreator<SyncVariablePluginOptions, FreeLayoutPluginContext>({
    onInit(ctx, options) {
      const flowDocument = ctx.document;

      // Listen for node creation events
      flowDocument.onNodeCreate(({ node }) => {
        const form = getNodeForm(node);
        const variableData = node.getData(FlowNodeVariableData);

        /**
         * Synchronizes output data to the variable engine.
         * @param value - The output data to synchronize.
         */
        const syncOutputs = (value: any) => {
          if (!value) {
            // If the output data is empty, clear the variable
            variableData.clearVar();
            return;
          }

          // Create an Type AST from the output data's JSON schema
          // NOTICE: You can create a new function to generate an AST based on YOUR CUSTOM DSL
          const typeAST = createASTFromJSONSchema(value);

          if (typeAST) {
            // Use the node's title or its ID as the title for the variable
            const title = form?.getValueIn('title') || node.id;

            // Set the variable in the variable engine
            variableData.setVar(
              ASTFactory.createVariableDeclaration({
                meta: {
                  title: `${title}`,
                  icon: node.getNodeRegistry()?.info?.icon,
                  // NOTICE: You can add more metadata here as needed
                },
                key: `${node.id}.outputs`,
                type: typeAST,
              })
            );
          } else {
            // If the AST cannot be created, clear the variable
            variableData.clearVar();
          }
        };

        if (form) {
          // Initially synchronize the output data
          syncOutputs(form.getValueIn('outputs'));

          // Listen for changes in the form values and re-synchronize when outputs change
          form.onFormValuesChange((props) => {
            if (props.name.match(/^outputs/) || props.name.match(/^title/)) {
              syncOutputs(form.getValueIn('outputs'));
            }
          });
        }
      });
    },
  });

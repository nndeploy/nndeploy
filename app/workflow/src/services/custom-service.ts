import { injectable, inject } from '@flowgram.ai/free-layout-editor';
import {
  FreeLayoutPluginContext,
  SelectionService,
  Playground,
  WorkflowDocument,
} from '@flowgram.ai/free-layout-editor';

/**
 * Docs: https://inversify.io/docs/introduction/getting-started/
 * Warning: Use decorator legacy
 *   // rsbuild.config.ts
 *   {
 *     source: {
 *       decorators: {
 *         version: 'legacy'
 *       }
 *     }
 *   }
 * Usage:
 *  1.
 *    const myService = useService(CustomService)
 *    myService.save()
 *  2.
 *    const myService = useClientContext().get(CustomService)
 *  3.
 *    const myService = node.getService(CustomService)
 */
@injectable()
export class CustomService {
  @inject(FreeLayoutPluginContext) ctx: FreeLayoutPluginContext;

  @inject(SelectionService) selectionService: SelectionService;

  @inject(Playground) playground: Playground;

  @inject(WorkflowDocument) document: WorkflowDocument;

  save() {
    console.log(this.document.toJSON());
  }
}

import { FormRenderProps, FlowNodeJSON } from '@flowgram.ai/free-layout-editor';
import { SubCanvasRender } from '@flowgram.ai/free-container-plugin';

import { FormHeader } from '../../form-header';
import { FormContent } from '../../../../../form-components';
import { useIsSidebar } from '../../../../../hooks';

export const ContainerFormRender = ({ form }: FormRenderProps<FlowNodeJSON>) => {
  const isSidebar = useIsSidebar();
  if (isSidebar) {
    return (
      <>
        <FormHeader />
        <FormContent>
          {/* <FormInputs />
          <FormOutputs /> */}
        </FormContent>
      </>
    );
  }
  return (
    <>
      <FormHeader />
      <FormContent>
        {/* <FormInputs /> */}
        <SubCanvasRender />
        {/* <FormOutputs /> */}
      </FormContent>
    </>
  );
};

import { Form, Button, Toast } from "@douyinfe/semi-ui";

export interface BranchEditDrawerProps {
  onSure: (node: any) => void;
  onClose: () => void;
}

const FlowConfigDrawer: React.FC<BranchEditDrawerProps> = (props) => {
  async function onSure() {
    try {
    
      Toast.success("save sucess!");
    } catch (error) {
      Toast.error("save fail " + error);
    }
  }
  return (
    <>
      <div className="drawer-content">
       </div> 
      <div className="semi-sidesheet-footer">
        <Button onClick={() => onSure()}>confirm</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
      </div>
    </>
  );
};

export default FlowConfigDrawer;

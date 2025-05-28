
import { ResourceTreeNodeData } from '../entity';
import './index.scss'
const ImagePreview: React.FC<ResourceTreeNodeData> = (props) => {
  return (
    <div className="image-preview">
      <img src={props.url} />
    </div>
  );
};

export default ImagePreview;

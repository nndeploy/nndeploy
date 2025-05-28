
import { VideoPlayer } from '@douyinfe/semi-ui';
import './index.scss'
import { ResourceTreeNodeData } from '../entity';
const VideoPreview: React.FC<ResourceTreeNodeData> = (props) => {
  return (
    <div className="video-preview">
      <VideoPlayer
            height={430} 
            src={props.url}
            //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
        />
    </div>
  );
};

export default VideoPreview;

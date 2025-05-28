
import { ResourceTreeNodeData } from "../entity";
import ImagePreview from "../ImagePreview";
import VideoPreview from "../VidoPreview";

const Preview: React.FC<ResourceTreeNodeData> = (props) => {
  return (
    <>
      {props.mime == "image" ? (
        <ImagePreview {...props} />
      ) : props.mime == "video" ? (
        <VideoPreview {...props} />
      ) : (
        <></>
      )}
    </>
  );
};

export default Preview;

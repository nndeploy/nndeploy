import IoTypeBinary from "./IoTypeBinary";
import IoTypeBoolean from "./IoTypeBoolean";
import IoTypeImage from "./IoTypeImage";
import IoTypeNumber from "./IoTypeNumber";
import IoTypeString from "./IoTypeString";
import IoTypeTextFile from "./IoTypeTextFile";
import IoTypeVideo from "./IoTypeVideo";

export enum EnumIODataType {
  Bool = 'Bool',
  Num = 'Num',
  String = 'String',
  Text = 'Text',
  Json = 'Json',
  Xml = 'Xml',
  Csv = 'Csv',
  Yaml = 'Yaml',
  Binary = 'Binary',
  Video = 'Video',
  Audio = 'Audio',
  Image = 'Image',
  Camera = 'Camera',
  Microphone = 'Microphone',
  Model = 'Model',
  Dir = 'Dir',
  Any = 'Any'

}
interface IoTypeProps {
 // value: string;
  direction: 'input' | 'output',
 // nodeName:string; 
  ioDataType: EnumIODataType,
  onChange: (value: any) => void;
  node:any
}

type ComponentMap = {
  [key: string]: React.ComponentType<any>;
};

const IoType: React.FC<IoTypeProps> = (props) => {
  const {  onChange, ioDataType, direction,
    //nodeName 
    node
  } = props;

  const COMPONENT_MAP: ComponentMap = {

    [EnumIODataType.String]: IoTypeString,
    [EnumIODataType.Json]: IoTypeString,
    [EnumIODataType.Xml]: IoTypeString,
    [EnumIODataType.Csv]: IoTypeString,
    [EnumIODataType.Yaml]: IoTypeString,

    [EnumIODataType.Text]: IoTypeTextFile,
    [EnumIODataType.Image]: IoTypeImage,

    [EnumIODataType.Video]: IoTypeVideo,
    [EnumIODataType.Binary]: IoTypeBinary,
    [EnumIODataType.Model]: IoTypeBinary,

    [EnumIODataType.Bool]: IoTypeBoolean,
    [EnumIODataType.Num]: IoTypeNumber,

  };

  if (!COMPONENT_MAP[ioDataType]) {

    return <></>
  }

  const Component = COMPONENT_MAP[ioDataType] ?? <></>;

  return (
    <Component 
    //key={ nodeName + '_' +  ioDataType } 
    onChange={onChange} direction={direction} ioDataType={ioDataType} node={node} />
  )
}

export default IoType
import { Button, Popover, SideSheet, Tag, Typography } from "@douyinfe/semi-ui";
import { useGetTemplates } from "../../../Home/effect";
import classNames from "classnames";
import { IconPlus } from "@douyinfe/semi-icons";
import { IWorkFlowShortEntity } from "../../../../entity";
import './index.scss'

const { Text, Paragraph, Title } = Typography;

interface ITemplateDrawerProps {
  visible: boolean;
  onCancel: () => void;
  rightContentRef: React.RefObject<HTMLDivElement>
  onAddTemplate: (item: IWorkFlowShortEntity) => void;
}

const TemplateDrawer: React.FC<ITemplateDrawerProps> = (props) => {

  const { templates, loading, error, getTemplates } = useGetTemplates();

  const { visible } = props

  function onAddToWorkspace(item: IWorkFlowShortEntity): void {
   
    props.onAddTemplate(item)
   // throw new Error("Function not implemented.");
  }



  return (
    <SideSheet visible={visible} onCancel={props.onCancel} placement="left" className={'templateDrawer'}

      width={'100%'}
      getPopupContainer={() => {

        return props.rightContentRef.current!
      }
      }
    >
      <div className="area-template">
        <Title heading={1} style={{ margin: '8px 0' }} >Template</Title>
        <Paragraph
          //type="secondary" 
          size="normal" style={{ fontSize: '16px' }} >Use the following template workflows, or customize your own workflows based on the templates.</Paragraph>


        <div className="items">

          {
            templates.map((item) => (
              <div className={classNames("item", { noCover: !item.cover_ })} key={item.id}>
                <div className="image-cover">
                  <img src={`/api/preview?file_path=${item.cover_}`} alt="" width={320} height={180} />
                </div>

                <div className={classNames("item-content")}>

                  <div className="title">{item.name_}</div>

                  <div className="developer">{item.developer_ ?? 'unknown developer'}</div>

                  <div className="desc">
                    {item.desc_ ?

                      <Popover content={item.desc_} className="desc-popover">

                        {item.desc_}
                      </Popover>

                      : 'No description'
                    }



                  </div>

                  <div className="source">


                    <a href={item.source_ ? item.source_.split(',')[0] : ''} target="_blank">
                      <Tag
                        color='light-blue'
                        //prefixIcon={<IconGithubLogo />}
                        size='large'
                        //shape='circle'
                        type='light'
                        style={{ maxWidth: '100%' }}

                      >

                        {item.source_ ? item.source_.split(',')[0] : 'unknown source'}

                      </Tag>
                    </a>

                  </div>

                  <div className="bottom">

                    <Button
                      icon={<IconPlus />}
                      size="small"
                      style={{ borderRadius: '8px' }}
                      block
                      theme='solid' type="primary" onClick={() => onAddToWorkspace(item)}>Add to Workspace</Button>


                  </div>
                </div>
              </div>
            ))

          }

        </div>
      </div>
    </SideSheet>
  );
};

export default TemplateDrawer;
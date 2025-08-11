import { useState } from "react";
import { JsonSchemaEditor } from "../components/json-schema-editor";
import { JsonSchema } from "../components/type-selector/types";
import './index.scss'
import { useGetTemplates, useGetWorkflows } from "./effect";
import { Button, Tag } from "@douyinfe/semi-ui";
import { IconPlus } from "@douyinfe/semi-icons";
import classNames from "classnames";
import Title from "@douyinfe/semi-ui/lib/es/typography/title";
import { Typography } from '@douyinfe/semi-ui';
import { useNavigate, useNavigation } from "react-router-dom";
import { IWorkFlowShortEntity } from "../../entity";
import Header from "../Layout/Design/header";
const { Text, Paragraph } = Typography;

export default function Home() {


  const navigate = useNavigate();

  const { templates, loading, error, getTemplates } = useGetTemplates();

  const { workFlows } = useGetWorkflows();

  function onCreateNewWorkFlow(){
    navigate('/design')
  }

  function onWorkFlowClick(item: IWorkFlowShortEntity){
    navigate(`/design?id=${item.id}&name=${item.name_}&flowType=workspace`)
  }

  async function onAddToWorkspace(item: IWorkFlowShortEntity){
   navigate(`/design?id=${item.id}&name=${item.name_}&flowType=template`)
  }

  return (
    <div className="home-page">
      <Header />
      <div className="areas">
        <div className="area-workspace">

          <Title heading={2} style={{ margin: '8px 0' }} >Workspace</Title>
          {/* <Text type="secondary" >Create a blank workflow</Text> */}


          <div className="items">
            <div className="add-item" onClick={()=>onCreateNewWorkFlow()}>
              <IconPlus size="extra-large" />

              <Paragraph type="secondary" >Click me to create a blank workflow</Paragraph>

            </div>
            {
              workFlows.map((item) => (
                <div className={classNames("item")} key={item.id} onClick={()=>onWorkFlowClick(item)}>



                  <div className={classNames("item-content")}>

                    <div className="title">{item.name_}</div>

                    <div className="desc">
                      {item.desc_ ?? 'No description'}

                    </div>

                  </div>
                </div>
              ))

            }

          </div>
        </div>
        <div className="area-template">
          <Title heading={2} style={{ margin: '8px 0' }} >Template</Title>
          <Text type="secondary" >Use the following template workflows, or customize your own workflows based on the templates.</Text>

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
                      {item.desc_ ?? 'No description'}

                    </div>
                    <div className="source">


                      <a href={item.source_ ? item.source_.split(',')[0]: ''} target="_blank">
                        <Tag
                          color='light-blue'
                          //prefixIcon={<IconGithubLogo />}
                          size='large'
                          //shape='circle'
                          type='light'
                        >

                          {item.source_ ? item.source_.split(',')[0].substr(0, 50) :  'unknown source'}

                        </Tag>
                      </a>

                    </div>

                    <div className="bottom">

                      <Button
                        icon={<IconPlus />}
                        size="small"
                        style={{ borderRadius: '8px' }}
                        block
                        theme='solid' type="primary" onClick={()=>onAddToWorkspace(item)}>Add to Workspace</Button>


                    </div>
                  </div>
                </div>
              ))

            }

          </div>
        </div>
      </div>

    </div >
  );
}

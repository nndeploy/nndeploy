import React, { useImperativeHandle, useRef, useState } from "react";
import {
  Nav,
  Tabs,
  TabPane,
  Tooltip,
  Avatar,
  Dropdown,
  Button,
} from "@douyinfe/semi-ui";
import {
  IconBranch,
  IconFile,
  IconPlus,
  IconUser,
  IconHelpCircle,
  IconGithubLogo,
  IconApps,
} from "@douyinfe/semi-icons";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faDiscord,
  faWeixin,
  faZhihu,
  
} from "@fortawesome/free-brands-svg-icons";


import { Typography } from "@douyinfe/semi-ui";
import "./Design.scss";
import "./FirstLevelNav.scss";
import "./SecondLevelNav.scss";
import "./TabContent.scss";
import Flow from "../../components/flow/Index";

import companyLogo from "../../../assets/kapybara_logo.png";
import NodeTree from "./Node";
import Resource from "./Resource";
import WorkFlow, { WorkFlowComponentHandle } from "./WorkFlow";
import { IResourceTreeNodeEntity } from "./Resource/entity";
import { IWorkFlowEntity } from "./WorkFlow/entity";

var tabId  = 1; 

const Design: React.FC = () => {
  const { Text } = Typography;

  const [activeKey, setActiveKey] = useState<string>("1");
   
  const [tabs, setTabs] = useState<Array<{ id: string, tabId:string, name: string; newName:string  }>>([
    //{ name: "Tab 1", id: "1" },
  ]);
  const [selectedFirstLevel, setSelectedFirstLevel] = useState<string | null>(
    null
  );

   const workFlowTreeRef = useRef<WorkFlowComponentHandle>(null);



  const handleFirstLevelClick = (key: string) => {
    if (selectedFirstLevel === key) {
      setSelectedFirstLevel(null);
    } else {
      setSelectedFirstLevel(key);
    }
  };

  const handleTabClose = (key: string) => {
    const newTabs = tabs.filter((tab) => tab.tabId !== key);
    setTabs(newTabs);
    if (activeKey === key && newTabs.length > 0) {
      setActiveKey(newTabs[0].tabId);
    }
  };

  const handleAddTab = () => {
    //const newKey = `${tabs.length + 1}`;
    const  newTabId = `${tabId++}` 
    setTabs( [...tabs, {  newName: `Unsaved Workflow ${newTabId}` , name: '' , id: '', tabId: newTabId}]);
    setActiveKey(newTabId);
  };

  function onShowFlow(node: IResourceTreeNodeEntity){
    const newTabId = `${tabId++}`
    setTabs( [...tabs, { name: node.name, id: node.id, tabId: newTabId, newName: ''}]);
    setActiveKey(newTabId);
  }

  function onFlowSave(flow: IWorkFlowEntity){
    const newTabs = tabs.map((tab) => {
      if(tab.tabId === activeKey){
        return { ...tab, name: flow.name, newName: ''}
      }
      return tab;
    })
    setTabs(newTabs);
    workFlowTreeRef.current?.refresh()
    //setActiveKey(activeKey);
  }

  return (
    <div className="container design-page">
      <Nav mode="horizontal" className="topNav" >
        <Nav.Header>
          <img
            src={companyLogo}
            width="100"
            alt="Logo"
            className="companyLogo" 
          />
        </Nav.Header>
        <Nav.Footer>
          <a href="https://github.com/nndeploy/nndeploy" target="_blank">
            <Button icon={<IconGithubLogo/>} theme="borderless"  size='large' />
          </a>

          <a
            href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"
            target="_blank"
          >
            <Button icon={<IconHelpCircle />} size='large' theme="borderless" />
          </a>
          <a
            href="https://www.zhihu.com/column/c_1690464325314240512"
            target="_blank"
          >
            <Button
              icon={<FontAwesomeIcon icon={faZhihu} size="1x" />}
              size='large'
              theme="borderless"
            />
          </a>
          <a href="https://discord.gg/xAWvmZn3" target="_blank">
            <Button
              icon={<FontAwesomeIcon icon={faDiscord} size="1x" />}
              size='large'
              theme="borderless"
            />
          </a>
          <a
            href="https://github.com/nndeploy/nndeploy/blob/main/docs/zh_cn/knowledge_shared/wechat.md"
            target="_blank"
          >
            <Button
              icon={<FontAwesomeIcon icon={faWeixin} size="1x" />}
              theme="borderless"
              size='large'
            />
          </a>
          <Dropdown
            render={
              <Dropdown.Menu>
                <Dropdown.Item>Profile</Dropdown.Item>
                <Dropdown.Item>Logout</Dropdown.Item>
              </Dropdown.Menu>
            }
          >
            <Avatar color="blue" size="small">
              <IconUser size="small" />
            </Avatar>
          </Dropdown>
        </Nav.Footer>
      </Nav>
      <div className="main">
        <div className="leftNav">
          <Nav
            mode="vertical"
            className="firstLevelNav"
            selectedKeys={[selectedFirstLevel!]}
          >
            <Tooltip content="Resources" position="right">
              <Nav.Item
                itemKey="resources"
                icon={<IconFile />}

                onClick={() => handleFirstLevelClick("resources")}
              />
            </Tooltip>

            <Tooltip content="Nodes" position="right">
              <Nav.Item
                itemKey="nodes"
                icon={<IconBranch />}

                onClick={() => handleFirstLevelClick("nodes")}
              />
            </Tooltip>
            <Tooltip content="Workflow" position="right">
              <Nav.Item
                itemKey="workflow"
                icon={<IconApps />}
              
                onClick={() => handleFirstLevelClick("workflow")}
              />
            </Tooltip>
          </Nav>
          {
            selectedFirstLevel === "nodes" ?  (
              <Nav mode="vertical" className="secondLevelNav">
                <NodeTree />
              </Nav>
            ): selectedFirstLevel === "resources" ?  (
              <Nav mode="vertical" className="secondLevelNav">
                <Resource />
              </Nav>
            ): selectedFirstLevel === "workflow" ?  (
              <Nav mode="vertical" className="secondLevelNav">
                <WorkFlow onShowFlow = {onShowFlow} ref={workFlowTreeRef}/>
              </Nav>
            ): <></>
          }
         
        </div>
        <div className="rightContent">
          <Tabs
            tabPosition="top"
            activeKey={activeKey}
            onChange={setActiveKey}
            size="small"
            className="tabs top-tabs"
            onTabClose={handleTabClose}
            tabBarExtraContent={
              <Tooltip content="add" position="top">
                <Text
                  link
                  icon={<IconPlus />}
                  onClick={() => handleAddTab()}
                ></Text>
              </Tooltip>
            }
          >
            {tabs.map((tab) => (
              <TabPane
                tab={tab.name ? tab.name: tab.newName}
                itemKey={tab.tabId}
                key={tab.tabId}
                closable={true}
               
              >
                <div className="tab-content">
                  <Flow id={tab.id}  onFlowSave={onFlowSave}/>
                </div>
              </TabPane>
            ))}
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Design;

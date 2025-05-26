import React, { useState, useEffect, useRef } from "react";
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
import "./Backend.scss";
import "./FirstLevelNav.scss";
import "./SecondLevelNav.scss";
import "./TabContent.scss";
import Flow from "../../components/flow/Index";
import { ItemKey } from "@douyinfe/semi-ui/lib/es/navigation/Item";

import companyLogo from "../../../assets/kapybara_logo.png";
import { apiGetFiles } from "./api";
import NodeTree from "./Node";

const Backend: React.FC = () => {
  const { Text } = Typography;

  const [activeKey, setActiveKey] = useState<string>("1");
  const [tabs, setTabs] = useState<Array<{ tab: string; key: string }>>([
    { tab: "Tab 1", key: "1" },
  ]);
  const [selectedFirstLevel, setSelectedFirstLevel] = useState<string | null>(
    null
  );

  const handleFirstLevelClick = (key: string) => {
    if (selectedFirstLevel === key) {
      setSelectedFirstLevel(null);
    } else {
      setSelectedFirstLevel(key);
    }
  };

  const handleTabClose = (key: string) => {
    const newTabs = tabs.filter((tab) => tab.key !== key);
    setTabs(newTabs);
    if (activeKey === key && newTabs.length > 0) {
      setActiveKey(newTabs[0].key);
    }
  };

  const handleAddTab = () => {
    const newKey = `${tabs.length + 1}`;
    setTabs([...tabs, { tab: `Tab ${newKey}`, key: newKey }]);
    setActiveKey(newKey);
  };

  //  const dropzone = useRef<HTMLElement | null>(null);
  
  //   useEffect(() => {
  //     if (dropzone.current) {
  
  //        dropzone.current.addEventListener('dragover', (e) => {
  //             e.preventDefault();
  //             //dropzone.classList.add('over'); // 鼠标经过目标区域时强调样式
  //         });
  
  //         // dropzone.current.addEventListener('dragleave', () => {
  //         //     //dropzone.classList.remove('over'); // 离开时恢复样式
  //         // });
  
  //       dropzone.current.addEventListener("drop", async(e) => {
  //         e.preventDefault();
  
  //       });
  //     }
  //   }, [dropzone.current]);


  function onDrapOver(e:any){
    e.preventDefault();
    var i = 0; 
  }

  function onDrop(e:any){
    

     const nodeType = e?.dataTransfer?.getData("text");

     var i = 0; 


  }


  return (
    <div className="container backend-page">
      <Nav mode="horizontal" className="topNav" >
        <Nav.Header>
          {/* <IconSemiLogo />
          <span className="companyName">公司名称</span> */}
          <img
            src={companyLogo}
            width="100"
            alt="Logo"
            className="companyLogo" 
            onDragOver={onDrapOver}
            onDrop={onDrop}
            ///@ts-ignore
            // ref={dropzone

            // }
          />
        </Nav.Header>
        <Nav.Footer>
          <a href="https://github.com/nndeploy/nndeploy" target="_blank">
            <Button icon={<IconGithubLogo />} theme="borderless" />
          </a>

          <a
            href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"
            target="_blank"
          >
            <Button icon={<IconHelpCircle />} theme="borderless" />
          </a>
          <a
            href="https://www.zhihu.com/column/c_1690464325314240512"
            target="_blank"
          >
            <Button
              icon={<FontAwesomeIcon icon={faZhihu} size="1x" />}
              theme="borderless"
            />
          </a>
          <a href="https://discord.gg/xAWvmZn3" target="_blank">
            <Button
              icon={<FontAwesomeIcon icon={faDiscord} size="1x" />}
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
              <IconUser />
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
                //className={selectedFirstLevel === "resources" ? "selected" : ""}
                onClick={() => handleFirstLevelClick("resources")}
              />
            </Tooltip>

            <Tooltip content="Nodes" position="right">
              <Nav.Item
                itemKey="nodes"
                icon={<IconBranch />}
                //className={selectedFirstLevel === "nodes" ? "selected" : ""}
                onClick={() => handleFirstLevelClick("nodes")}
              />
            </Tooltip>
            <Tooltip content="Workflow" position="right">
              <Nav.Item
                itemKey="workflow"
                icon={<IconApps />}
                //className={selectedFirstLevel === "workflow" ? "selected" : ""}
                onClick={() => handleFirstLevelClick("workflow")}
              />
            </Tooltip>
          </Nav>
          {
            selectedFirstLevel === "nodes" ?  (
              <Nav mode="vertical" className="secondLevelNav">
                <NodeTree />
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
                tab={tab.tab}
                itemKey={tab.key}
                key={tab.key}
                closable={true}
              >
                <div className="tab-content">
                  <Flow />
                </div>
              </TabPane>
            ))}
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Backend;

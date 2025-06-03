import { Link, Outlet } from "react-router-dom";
import React, { useState } from "react";
import {
  Nav,
  Avatar,
  Tabs,
  TabPane,
  Dropdown,
  Button,
} from "@douyinfe/semi-ui";
import {
  IconSemiLogo,
  IconFeishuLogo,
  IconHelpCircle,
  IconBell,
  IconUser,
  IconGithubLogo,
} from "@douyinfe/semi-icons";
//import * as Icons from "@douyinfe/semi-icons-lab";
import * as Icons from "@douyinfe/semi-icons";

import styles from "./index.module.scss";
import "./index.scss";
import { menuConfig } from "./menuConfig";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faDiscord,
  faWeixin,
  faZhihu,
} from "@fortawesome/free-brands-svg-icons";

import companyLogo from "../../../assets/kapybara_logo.png";

const BackendLayout: React.FC<any> = (props: any) => {

 // debugger;
  const [activeKey, setActiveKey] = useState<string>("1");
  const [tabs, setTabs] = useState<
    Array<{ tab: string; key: string; url: string }>
  >([{ tab: "Home", key: "1", url: "https://example.com/home" }]);

  const handleMenuClick = (key: string, text: string, url: string) => {
    const existingTab = tabs.find((tab) => tab.key === key);
    if (!existingTab) {
      setTabs([...tabs, { tab: text, key, url }]);
    }
    setActiveKey(key);
  };

  const handleTabClose = (key: string) => {
    const newTabs = tabs.filter((tab) => tab.key !== key);
    setTabs(newTabs);
    if (activeKey === key && newTabs.length > 0) {
      setActiveKey(newTabs[0].key);
    }
  };

  const renderMenuItems = (items: any[]) => {
    return items.map((item) => (
      <Nav.Item
        key={item.key}
        itemKey={item.key}
        text={item.text}
        ///@ts-ignore
        icon={React.createElement(Icons[item.icon], {
          className: styles.iconIntro,
        })}
        className={styles.navItem}
        onClick={() => handleMenuClick(item.key, item.text, item.url)}
      />
    ));
  };

  //return <h2>backend</h2>
  const renderMenu = () => {
    return menuConfig.map((item) => {
      if (item.items) {
        return (
          <Nav.Sub
            key={item.key}
            itemKey={item.key}
            text={item.text}
            ///@ts-ignore
            icon={React.createElement(Icons[item.icon])}
          >
            {renderMenuItems(item.items)}
          </Nav.Sub>
        );
      } else {
        return (
          <Nav.Item
            key={item.key}
            itemKey={item.key}
            text={item.text}
            ///@ts-ignore
            icon={React.createElement(Icons[item.icon], {
              
            })}
          
            onClick={() => handleMenuClick(item.key, item.text, item.url!)}
          />
        );
      }
    });
  };

  return <div className="container backend-page">
      <Nav mode="horizontal" className="topNav">
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
            <Button icon={<IconGithubLogo />} theme="borderless" size="large" />
          </a>

          <a
            href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"
            target="_blank"
          >
            <Button icon={<IconHelpCircle />} size="large" theme="borderless" />
          </a>
          <a
            href="https://www.zhihu.com/column/c_1690464325314240512"
            target="_blank"
          >
            <Button
              icon={<FontAwesomeIcon icon={faZhihu} size="1x" />}
              size="large"
              theme="borderless"
            />
          </a>
          <a href="https://discord.gg/xAWvmZn3" target="_blank">
            <Button
              icon={<FontAwesomeIcon icon={faDiscord} size="1x" />}
              size="large"
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
              size="large"
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
        <Nav
          defaultOpenKeys={["user", "union"]}
          bodyStyle={{ height: 918 }}
          mode="vertical"
          footer={{ collapseButton: true }}
          className="left"
        >
          {renderMenu()}
        </Nav>
        {/* <div className="my">
          <div className="content">content.................</div>
        </div> */}
        <div className="mainRight">
          <Tabs
            tabPosition="top"
            activeKey={activeKey}
            onChange={setActiveKey}
            size="small"
            className="tabs"
            onTabClose={handleTabClose}
            // style={{ width: '100%',}}
            collapsible

            //overflow="scroll"
          >
            {tabs.map((tab) => (
              <TabPane
                tab={tab.tab}
                itemKey={tab.key}
                key={tab.key}
                closable={true}
              >
                <iframe src={tab.url} className="iframe" />
              </TabPane>
            ))}
          </Tabs>
        </div>
      </div>
    </div>
  
};
export default BackendLayout;

import { Link, Outlet } from "react-router-dom";
import React, { useState } from 'react';
import { Nav, Avatar, Tabs, TabPane, Dropdown } from '@douyinfe/semi-ui';
import { IconSemiLogo, IconFeishuLogo, IconHelpCircle, IconBell } from '@douyinfe/semi-icons';
import * as Icons  from '@douyinfe/semi-icons-lab';

import styles from './index.module.scss';
import  './index.scss';
import menuConfig from './menuConfig.json';

const HomeLayout: React.FC<any> = (props: any) => {

  const [activeKey, setActiveKey] = useState<string>('1');
  const [tabs, setTabs] = useState<Array<{ tab: string, key: string, url: string }>>([
    { tab: 'Home', key: '1', url: 'https://example.com/home' }
  ]);

  const handleMenuClick = (key: string, text: string, url: string) => {
    const existingTab = tabs.find(tab => tab.key === key);
    if (!existingTab) {
      setTabs([...tabs, { tab: text, key, url }]);
    }
    setActiveKey(key);
  };

  const handleTabClose = (key: string) => {
    const newTabs = tabs.filter(tab => tab.key !== key);
    setTabs(newTabs);
    if (activeKey === key && newTabs.length > 0) {
      setActiveKey(newTabs[0].key);
    }
  };

  const renderMenuItems = (items: any[]) => {
    return items.map(item => (
      <Nav.Item
        key={item.key}
        itemKey={item.key}
        text={item.text}
        ///@ts-ignore
        icon={React.createElement(Icons[item.icon], { className: styles.iconIntro })}
        className={styles.navItem}
        onClick={() => handleMenuClick(item.key, item.text, item.url)}
      />
    ));
  };

  const renderMenu = () => {
    return menuConfig.map(group => (
      <Nav.Sub
        key={group.key}
        itemKey={group.key}
        text={group.text}
         ///@ts-ignore
        icon={React.createElement(Icons[group.icon], { className: styles.iconIntro })}
      >
        {renderMenuItems(group.items)}
      </Nav.Sub>
    ));
  };

 
  return (
    <div className={styles.frame}>
      <Nav
        mode="horizontal"
        header={{
          logo: <IconSemiLogo className={styles.semiIconsSemiLogo} />,
          text: "Semi Templates",
        }}
        footer={
          <div className={styles.dIv}>
            <IconFeishuLogo size="large" className={styles.semiIconsFeishuLogo} />
            <IconHelpCircle size="large" className={styles.semiIconsFeishuLogo} />
            <IconBell size="large" className={styles.semiIconsFeishuLogo} />
            <Avatar
              size="small"
              src="https://sf6-cdn-tos.douyinstatic.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/root-web-sites/avatarDemo.jpeg"
              color="blue"
              className={styles.avatar}
            >
              示例
            </Avatar>
          </div>
        }
        className={styles.nav}
      >
        {/* <Nav.Item itemKey="Home" text="Home" />
        <Nav.Item itemKey="Project" text="Project" />
        <Nav.Item itemKey="Board" text="Board" />
        <Nav.Item itemKey="Forms" text="Forms" /> */}
      </Nav>
      <div className={styles.main}>
        <Nav
          defaultOpenKeys={["user", "union"]}
          bodyStyle={{ height: 918 }}
          mode="vertical"
          footer={{ collapseButton: true }}
          className={styles.left}
        >
          {renderMenu()}
        </Nav>
        {/* <div className="my">
          <div className="content">content.................</div>
        </div> */}
        <div className={styles.mainRight}>
          <Tabs
            tabPosition="top"
            activeKey={activeKey}
            onChange={setActiveKey}
            size="small"
            className={styles.tabs}
            
            onTabClose={handleTabClose}
            // style={{ width: '100%',}}
            collapsible
       
            //overflow="scroll"
          >
            {tabs.map(tab => (
              <TabPane tab={tab.tab} itemKey={tab.key} key={tab.key} closable={true}>
                <iframe src={tab.url} className={styles.iframe} />
              </TabPane>
            ))}
          </Tabs>
         
        </div>
      </div>
    </div>
  );
}
export default HomeLayout;
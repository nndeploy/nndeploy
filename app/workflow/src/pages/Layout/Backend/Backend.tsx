import React, { useState, useEffect } from "react";
import {
  Nav,
  Tabs,
  TabPane,
  IconButton,
  Tooltip,
  Tree,
  Avatar,
  Dropdown,
  Button,
  ButtonGroup,
} from "@douyinfe/semi-ui";
import {
  IconCode,
  IconBranch,
  IconFile,
  IconDelete,
  IconPlus,
  IconUser,
  IconFeishuLogo,
  IconHelpCircle,
  IconBell,
  IconSemiLogo,
} from "@douyinfe/semi-icons";
import "./Backend.scss";
import "./FirstLevelNav.scss";
import "./SecondLevelNav.scss";
import "./TabContent.scss";
import Flow from "../../components/flow/Index";
import {
  ItemKey,
  SelectedData,
} from "@douyinfe/semi-ui/lib/es/navigation/Item";

const Backend: React.FC = () => {
  const [activeKey, setActiveKey] = useState<string>("1");
  const [tabs, setTabs] = useState<Array<{ tab: string; key: string }>>([
    { tab: "Tab 1", key: "1" },
  ]);
  const [selectedFirstLevel, setSelectedFirstLevel] = useState<string | null>(
    null
  );
  const [secondLevelData, setSecondLevelData] = useState<any[]>([]);

  const [selectedKeys, setSelectedKeys] = useState<ItemKey[]>([]);

  useEffect(() => {
    if (selectedFirstLevel) {
      if (selectedFirstLevel === "nodes") {
        fetchNodesData();
      } else if (selectedFirstLevel === "workflow") {
        fetchWorkflowData();
      } else if (selectedFirstLevel === "resources") {
        fetchResourcesData();
      }
    }
  }, [selectedFirstLevel]);

  const fetchNodesData = async () => {
    // 模拟ajax请求，返回三层级数据
    const response = await new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          {
            key: "1",
            label: "Node 1",
            children: [
              {
                key: "1-1",
                label: "Node 1-1",
                children: [
                  { key: "1-1-1", label: "Node 1-1-1" },
                  { key: "1-1-2", label: "Node 1-1-2" },
                ],
              },
              {
                key: "1-2",
                label: "Node 1-2",
                children: [
                  { key: "1-2-1", label: "Node 1-2-1" },
                  { key: "1-2-2", label: "Node 1-2-2" },
                ],
              },
            ],
          },
          {
            key: "2",
            label: "Node 2",
            children: [
              {
                key: "2-1",
                label: "Node 2-1",
                children: [
                  { key: "2-1-1", label: "Node 2-1-1" },
                  { key: "2-1-2", label: "Node 2-1-2" },
                ],
              },
              {
                key: "2-2",
                label: "Node 2-2",
                children: [
                  { key: "2-2-1", label: "Node 2-2-1" },
                  { key: "2-2-2", label: "Node 2-2-2" },
                ],
              },
            ],
          },
        ]);
      }, 200);
    });
    setSecondLevelData(response);
  };

  const fetchWorkflowData = async () => {
    // 模拟ajax请求，返回三层级数据
    const response = await new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          {
            key: "1",
            label: "Workflow 1",
            children: [
              {
                key: "1-1",
                label: "Workflow 1-1",
                children: [
                  { key: "1-1-1", label: "Workflow 1-1-1" },
                  { key: "1-1-2", label: "Workflow 1-1-2" },
                ],
              },
              {
                key: "1-2",
                label: "Workflow 1-2",
                children: [
                  { key: "1-2-1", label: "Workflow 1-2-1" },
                  { key: "1-2-2", label: "Workflow 1-2-2" },
                ],
              },
            ],
          },
          {
            key: "2",
            label: "Workflow 2",
            children: [
              {
                key: "2-1",
                label: "Workflow 2-1",
                children: [
                  { key: "2-1-1", label: "Workflow 2-1-1" },
                  { key: "2-1-2", label: "Workflow 2-1-2" },
                ],
              },
              {
                key: "2-2",
                label: "Workflow 2-2",
                children: [
                  { key: "2-2-1", label: "Workflow 2-2-1" },
                  { key: "2-2-2", label: "Workflow 2-2-2" },
                ],
              },
            ],
          },
        ]);
      }, 200);
    });
    setSecondLevelData(response);
  };

  const fetchResourcesData = async () => {
    // 模拟ajax请求，返回三层级数据
    const response = await new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          {
            key: "1",
            label: "Resource 1",
            children: [
              {
                key: "1-1",
                label: "Resource 1-1",
                children: [
                  { key: "1-1-1", label: "Resource 1-1-1" },
                  { key: "1-1-2", label: "Resource 1-1-2" },
                ],
              },
              {
                key: "1-2",
                label: "Resource 1-2",
                children: [
                  { key: "1-2-1", label: "Resource 1-2-1" },
                  { key: "1-2-2", label: "Resource 1-2-2" },
                ],
              },
            ],
          },
          {
            key: "2",
            label: "Resource 2",
            children: [
              {
                key: "2-1",
                label: "Resource 2-1",
                children: [
                  { key: "2-1-1", label: "Resource 2-1-1" },
                  { key: "2-1-2", label: "Resource 2-1-2" },
                ],
              },
              {
                key: "2-2",
                label: "Resource 2-2",
                children: [
                  { key: "2-2-1", label: "Resource 2-2-1" },
                  { key: "2-2-2", label: "Resource 2-2-2" },
                ],
              },
            ],
          },
        ]);
      }, 200);
    });
    setSecondLevelData(response);
  };

  const handleFirstLevelClick = (key: string) => {
    if (selectedFirstLevel === key) {
      setSelectedFirstLevel(null);
      setSecondLevelData([]);
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

  return (
    <div className="container backend-page">
      <Nav mode="horizontal" className="topNav">
        <Nav.Header>
          <IconSemiLogo />
          <span className="companyName">公司名称</span>
        </Nav.Header>
        <Nav.Footer>
          <Button icon={<IconFeishuLogo />} theme="borderless" />
          <Button icon={<IconHelpCircle />} theme="borderless" />
          <Button icon={<IconBell />} theme="borderless" />
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
            <Tooltip content="Resources" position = "right" >
              <Nav.Item
                itemKey="resources"
                icon={<IconFile />}
                //className={selectedFirstLevel === "resources" ? "selected" : ""}
                onClick={() =>
                  handleFirstLevelClick("resources")
                }
              />
            </Tooltip>

            <Tooltip content="Nodes" position = "right" >
              <Nav.Item
                itemKey="nodes"
                icon={<IconCode />}
                //className={selectedFirstLevel === "nodes" ? "selected" : ""}
                onClick={() => handleFirstLevelClick("nodes")}
              />
            </Tooltip>
            <Tooltip content="Workflow" position = "right" >
              <Nav.Item
                itemKey="workflow"
                icon={<IconBranch />}
                //className={selectedFirstLevel === "workflow" ? "selected" : ""}
                onClick={() => handleFirstLevelClick("workflow")}
              />
            </Tooltip>
          </Nav>
          {selectedFirstLevel && (
            <Nav mode="vertical" className="secondLevelNav">
              <Tree treeData={secondLevelData} />
            </Nav>
          )}
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
              <ButtonGroup>
                <Button onClick={() => handleAddTab()}>新增</Button>
              </ButtonGroup>
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

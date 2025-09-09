import React, { useEffect, useReducer, useRef, useState } from "react";
import {
  Nav,
  Tabs,
  TabPane,
  Tooltip,
} from "@douyinfe/semi-ui";
import {
  IconBranch,
  IconFile,
  IconPlus,
  IconApps,
  IconInherit,
  IconArticle,
  IconPuzzle,
  IconAppCenter,
  IconCopyAdd,
} from "@douyinfe/semi-icons";



import { Typography } from "@douyinfe/semi-ui";
import "./Design.scss";
import "./FirstLevelNav.scss";
import "./SecondLevelNav.scss";
import "./TabContent.scss";
import Flow from "../../components/flow/Index";

import NodeTree from "./Node";
import Resource from "./Resource";
import WorkFlow from "./WorkFlow";
import { IWorkFlowEntity } from "./WorkFlow/entity";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { apiGetDagInfo } from "./api";
import store, { initialState, reducer } from "./store/store";
import { initDagGraphInfo } from "./store/actionType";
import { EnumFlowType } from "../../../enum";
import Header from "./header";
import TemplateDrawer from "./TemplateDrawer";
import { IWorkFlowShortEntity } from "../../../entity";
import { set } from "lodash";

var tabId = 2;

const Design: React.FC = () => {
  const { Text } = Typography;

  const [activeKey, setActiveKey] = useState<string>("1");


  // const search = window.location.search; // "?foo=bar&baz=qux"
  // const params = new URLSearchParams(search);

  const [tabs, setTabs] = useState<Array<{
    id: string, tabId: string, name: string;
    //newName: string, 
    flowType: EnumFlowType
  }>>([

    // { newName: "Unsaved Workflow 1", id: '1', name: '', tabId: '1' },
  ]);

  const [templateDrawerVisible, setTemplateDrawerVisible] = useState(false);

  function getQueryObject(url?: string) {
    const search = url ? url.split('?')[1] : window.location.search.slice(1);
    const params = new URLSearchParams(search);
    const obj: { [key: string]: string } = {};
    for (const [key, value] of params.entries()) {
      obj[key] = value;
    }
    return obj;
  }

  const params = getQueryObject()

  useEffect(() => {
    const newTabId = `${tabId++}`

    if (params.id && params.flowType) {

      setTabs([...tabs, {
        //newName: params.name, 
        name: params.name, id: params.id, tabId: newTabId, flowType: params.flowType as EnumFlowType
      }]);
      setActiveKey(newTabId);
    } else {

      setTabs([{
        //newName: "Unsaved Workflow 1", 
        id: '', name: 'Unsaved Workflow 1', tabId: '1', flowType: EnumFlowType.workspace
      }])


    }

  }, [])


  function onAddTemplate(item: IWorkFlowShortEntity) {

    setTemplateDrawerVisible(false)
    setSelectedFirstLevel('')

    const newTabId = `${tabId++}`

    setTabs([...tabs, {
      //newName: params.name, 
      name: item.name_, id: item.id, tabId: newTabId, flowType: EnumFlowType.template
    }]);
    setActiveKey(newTabId);
  }



  const [selectedFirstLevel, setSelectedFirstLevel] = useState<string | null>(
    null
  );

  const [state, dispatch] = useReducer(reducer, (initialState))

  async function getDagInfo() {
    var response = await apiGetDagInfo()
    if (response.flag != 'success') {
      return
    }

    dispatch(initDagGraphInfo(response.result))

  }

  useEffect(() => {
    getDagInfo()
  }, [])

  // const workFlowTreeRef = useRef<WorkFlowComponentHandle>(null);



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
    const newTabId = `${tabId++}`
    setTabs([...tabs, {
      // newName: `Unsaved Workflow ${newTabId}`, 
      name: `Unsaved Workflow ${newTabId}`, id: '', tabId: newTabId, flowType: EnumFlowType.workspace
    }]);

    setActiveKey(newTabId);
  };

  function onFlowDeleteCallBack(flowId: string) {

    const flow = tabs.find(item => item.id == flowId)
    const newTabs = tabs.filter(item => item.id != flowId)
    if (flow && flow.tabId == activeKey) {
      if (newTabs.length > 0) {
        setActiveKey(newTabs[0].tabId)
      }

    }

    setTabs(newTabs)



  }

  function onShowFlow(node: TreeNodeData) {

    const findTab = tabs.find(item => item.id == node.key as string) 
    if(findTab){
      setActiveKey(findTab.tabId)
      return
    }
  
    const newTabId = `${tabId++}`
    setTabs([...tabs, {
      name: node.label as string, id: node.key as string, tabId: newTabId,
      // newName: '',
      flowType: EnumFlowType.workspace
    }]);
    setActiveKey(newTabId);
  }

  function onFlowSave(flow: IWorkFlowEntity) {
    const newTabs = tabs.map((tab) => {
      if (tab.tabId === activeKey) {
        return {
          ...tab, name: flow.businessContent.name_, id: flow.id, flowType: EnumFlowType.workspace

          //newName: '' 
        }
      }
      return tab;
    })
    setTabs(newTabs);
    //workFlowTreeRef.current?.refresh()
    //setActiveKey(activeKey);
  }

  const [secondNavWith, setSecondNavWith] = useState(240)



  const [isDragging, setIsDragging] = useState(false); // 菜单拖拽

  const mouseRef = useRef<{ x: number, y: number }>({ x: 0, y: 0 }); // 鼠标信息
  function handleDrag(e: React.MouseEvent<HTMLDivElement>) {
    setIsDragging(true);

    mouseRef.current = {
      x: e.clientX,
      y: e.clientY,
    };
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  function handleMouseMove(e: MouseEvent) {
    if (isDragging) {
      const dx = e.clientX - mouseRef.current.x;
      const newWidth = secondNavWith + dx;
      setSecondNavWith(newWidth);
    }
  };

  // 处理鼠标释放
  function handleMouseUp() {
    if (isDragging) {
      setIsDragging(false);
    }
  };


  const rightContentRef = useRef<HTMLDivElement>(null)

  return (
    <div className="container design-page">
      <store.Provider value={{ state, dispatch }} >

        <Header />
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
                  icon={<IconArticle />}

                  onClick={() => handleFirstLevelClick("resources")}
                />
              </Tooltip>

              <Tooltip content="Nodes" position="right">
                <Nav.Item
                  itemKey="nodes"
                  icon={<IconPuzzle />}

                  onClick={() => handleFirstLevelClick("nodes")}
                />
              </Tooltip>
              <Tooltip content="Workspace" position="right">
                <Nav.Item
                  itemKey="workflow"
                  icon={<IconAppCenter />}

                  onClick={() => handleFirstLevelClick("workflow")}
                />
              </Tooltip>
              <Tooltip content="Template" position="right">
                <Nav.Item
                  itemKey="template"
                  icon={<IconCopyAdd />}

                  onClick={() => {
                    setTemplateDrawerVisible(true)
                  }}
                />
              </Tooltip>


            </Nav>
            {

              selectedFirstLevel ?
                <div className="secondLevelNav" style={{ width: `${secondNavWith}px` }}>
                  <Nav mode="vertical" >
                    {selectedFirstLevel === "nodes" ? <NodeTree />
                      : selectedFirstLevel === "resources" ? <Resource />
                        : selectedFirstLevel === "workflow" ? <WorkFlow onShowFlow={onShowFlow}
                          //ref={workFlowTreeRef} 
                          onFlowDeleteCallBack={onFlowDeleteCallBack} />
                          : <></>
                    }

                  </Nav>
                  <div className="second-level-dragger" onMouseDown={e => handleDrag(e)}></div>
                </div>
                : <></>
            }



          </div>
          <div className="rightContent" ref={rightContentRef}>
            <Tabs
              tabPosition="top"
              activeKey={activeKey}
              onChange={setActiveKey}
              size="small"
              className="tabs top-tabs"
              onTabClose={handleTabClose}
              tabPaneMotion={false}

              keepDOM={true}
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
                  tab={tab.name}
                  itemKey={tab.tabId}
                  key={tab.tabId}
                  closable={true}

                >
                  <div className="tab-content">
                    <Flow id={tab.id} onFlowSave={onFlowSave} activeKey={activeKey} flowType={tab.flowType} />

                  </div>
                </TabPane>
              ))}
            </Tabs>
          </div>
        </div>
        <TemplateDrawer visible={templateDrawerVisible} onCancel={() => setTemplateDrawerVisible(false)} rightContentRef={rightContentRef}

          onAddTemplate={onAddTemplate}
        />
      </store.Provider>
    </div >
  );
};

export default Design;

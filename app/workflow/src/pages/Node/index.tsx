import React, { useState, useEffect } from "react";
import {
  Form,
  Input,
  Button,
  Table,
  Pagination,
  SideSheet,
  IconButton,
} from "@douyinfe/semi-ui";
import {
  IconSearch,
  IconPlus,
  IconEyeClosed,
  IconEdit,
  IconDelete,
  IconFlowChartStroked,
} from "@douyinfe/semi-icons";
import "./index.scss";
import Flow from "../components/flow/Index";
import NodeTree from "./Tree";
import { INodeEntity } from "./entity";
import NodeEditDrawer from "./NodeEditDrawer";

const NodePage: React.FC = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  const [nodeEditVisible, setNodeEditVisible] = useState(false);
  const [nodeEdit, setNodeEdit] = useState<INodeEntity>();

  function onNodeEdit(item: INodeEntity) {
    setNodeEdit(item);
    setNodeEditVisible(true);
  }

  function onNodeEditDrawerSure(node: INodeEntity) {
    setNodeEditVisible(false);
  }

  function onNodeEditDrawerClose() {
    setNodeEditVisible(false);
  }

  const fetchData = async (params = {}) => {
    setLoading(true);
    // 模拟后台请求数据
    const response = await new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          data: Array.from({ length: 100 }, (_, index) => ({
            key: index + 1,
            name: `Name ${index + 1}`,
            age: 20 + (index % 10),
            address: `Address ${index + 1}`,
          })),
          total: 100,
        });
      }, 1000);
    });
    setData(response.data);
    setTotal(response.total);
    setLoading(false);
  };

  useEffect(() => {
    fetchData({ page: currentPage, pageSize });
  }, [currentPage, pageSize]);

  const handleSearch = (values: any) => {
    fetchData({ ...values, page: currentPage, pageSize });
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
  };

  function handleAdd() {
    onNodeEdit({ id: "", name: "", parentId: "", config: [] });
    setNodeEditVisible(true);
  }

  function handleEdit(entity: INodeEntity) {
    onNodeEdit(entity);
    setNodeEditVisible(true);
  }

  async function handleDelete(entity: INodeEntity){
    const response = await apiNodeDelete(entity)
    if(response.flag == "success"){
      
    }
  }

  const columns = [
    { title: "Name", dataIndex: "name", key: "name" },
    { title: "Age", dataIndex: "age", key: "age" },
    { title: "Address", dataIndex: "address", key: "address" },
    {
      title: "Actions",
      key: "actions",
      width: 160,
      align: "center" as const,
      render: (text: any, record: any) => (
        <>
          <IconButton
            icon={<IconEyeClosed />}
            onClick={() => console.log("View record:", record)}
          />
          <IconButton icon={<IconEdit />} onClick={() => handleEdit(record)} />
          <IconButton
            icon={<IconDelete />}
            onClick={() => handleDelete(record)}
          />
        </>
      ),
    },
  ];

  return (
    <div className="page-node">
      <NodeTree />
      <div className="main">
        <Form
          ///@ts-ignore
          layout="horizontal"
          onSubmit={handleSearch}
          className={"form"}
          labelPosition="inset"
        >
          <Form.Input field="name" label="Name" />
          <Form.Input field="age" label="Age" />
          <Button icon={<IconSearch />} htmlType="submit">
            Search
          </Button>
        </Form>
        <Button icon={<IconPlus />} onClick={handleAdd} className={"addButton"}>
          Add
        </Button>
        <Table
          columns={columns}
          dataSource={data.slice(
            (currentPage - 1) * pageSize,
            currentPage * pageSize
          )}
          loading={loading}
          pagination={false}
        />
        <Pagination
          total={total}
          currentPage={currentPage}
          pageSize={pageSize}
          onPageChange={handlePageChange}
          onPageSizeChange={handlePageSizeChange}
        />

        <SideSheet
          width={"calc(100% - 200px - 17px )"}
          visible={nodeEditVisible}
          onCancel={onNodeEditDrawerClose}
          title={nodeEdit?.name ?? "add"}
        >
          <NodeEditDrawer
            entity={nodeEdit!}
            onSure={onNodeEditDrawerSure}
            onClose={onNodeEditDrawerClose}
          />
        </SideSheet>
      </div>
    </div>
  );
};

export default NodePage;

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
import { apiGetNodePage, apiNodeDelete } from "./api";

const NodePage: React.FC = () => {
  const [data, setData] = useState<INodeEntity[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  const [parentId, setParentId] = useState("");

  const [nodeEditVisible, setNodeEditVisible] = useState(false);
  const [nodeEdit, setNodeEdit] = useState<INodeEntity>();

  function onNodeEdit(item: INodeEntity) {
    setNodeEdit(item);
    setNodeEditVisible(true);
  }

  function onNodeEditDrawerSure(node: INodeEntity) {
    setNodeEditVisible(false);
    getNodePage({ parentId, currentPage, pageSize });
  }

  function onNodeEditDrawerClose() {
    setNodeEditVisible(false);
  }

  async function getNodePage(query: any) {
    setLoading(true);

    const response = await apiGetNodePage(query);
    setData(response.result.records);
    setTotal(response.result.total);
    setLoading(false);
  }

  useEffect(() => {
    getNodePage({ parentId, currentPage, pageSize });
  }, [currentPage, pageSize]);

  const handleSearch = (values: any) => {
    getNodePage({ ...values, currentPage, pageSize, parentId });
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
  };

  function handleAdd() {
    onNodeEdit({
      id: "",
      name: "",
      parentId: "",
      schema: {
        type: "object",
        properties: {},
        required: [],
      },
    });
    setNodeEditVisible(true);
  }

  function handleEdit(entity: INodeEntity) {
    onNodeEdit(entity);
    setNodeEditVisible(true);
  }

  function onTreeSelect(key: string) {
    setParentId(key);
  }

  async function handleDelete(entity: INodeEntity) {
    const response = await apiNodeDelete(entity.id);
    if (response.flag == "success") {
      getNodePage({ currentPage, pageSize });
    }
  }

  const columns = [
    { title: "Name", dataIndex: "name", key: "name" },
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
      <NodeTree
        onSelect={function (key: string): void {
          onTreeSelect(key);
        }}
      />
      <div className="main">
        {/* <Form
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
        </Button> */}
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

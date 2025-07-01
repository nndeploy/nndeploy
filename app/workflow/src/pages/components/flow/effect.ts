import { useEffect, useState } from "react"
import { getNodeRegistry } from "./api"
import { FlowNodeRegistry } from "../../../typings"
import { apiGetNodeList } from "../../Layout/Design/Node/api"
import { INodeEntity } from "../../Node/entity"

export function useGetRegistry() {

  const [nodeRegistries, setNodeRegistries] = useState<FlowNodeRegistry[]>([])

  useEffect(() => {
    getNodeRegistry().then((res) => {
      setNodeRegistries(res)
    })
  }, [])

  return nodeRegistries
}

export  function useGetNodeList() {

  const [nodeList, setNodeList] = useState<INodeEntity[]>([])

  async function getNodeList() {
    const response = await apiGetNodeList()
    if (response.flag == 'success') {
      setNodeList(response.result)
    }
  }

  useEffect(() => {
    getNodeList()
  }, [])


  return nodeList
}

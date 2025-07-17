import { useEffect, useState } from "react"
import { apiGetParamTypes, getNodeRegistry } from "./api"
import { FlowNodeRegistry } from "../../../typings"
import { apiGetNodeList } from "../../Layout/Design/Node/api"
import { INodeEntity } from "../../Node/entity"
import { IParamTypes } from "../../Layout/Design/WorkFlow/entity"

export function useGetRegistry() {

  const [nodeRegistries, setNodeRegistries] = useState<FlowNodeRegistry[]>([])

  useEffect(() => {
    getNodeRegistry().then((res) => {
      setNodeRegistries(res)
    })
  }, [])

  return nodeRegistries
}

export function useGetNodeList() {

  const [nodeList, setNodeList] = useState<INodeEntity[]>([])

  async function getNodeList() {
    const response = await apiGetNodeList()
    if (response.flag == 'success') {

      const nodes = response.result.filter((item) => {
        return item.type == 'leaf'
      }).map(item => {
        return item.nodeEntity!
      })
      setNodeList(nodes)
    }
  }

  useEffect(() => {
    getNodeList()
  }, [])


  return nodeList
}

export function useGetParamTypes() {
  const [paramTypes, setParamTypes] = useState<IParamTypes>({});

  useEffect(() => {
    apiGetParamTypes().then(response => {
      if (response.flag == 'success') {
        setParamTypes(response.result)
      }
    })
  }, [])

  return paramTypes
}

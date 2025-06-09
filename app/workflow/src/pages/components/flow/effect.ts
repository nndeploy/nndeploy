import { useEffect, useState } from "react"
import { getNodeRegistry } from "./api"
import { FlowNodeRegistry } from "../../../typings"

export function useGetRegistry(){

  const [nodeRegistries, setNodeRegistries] = useState<FlowNodeRegistry[]>([])

  useEffect(() => {
    getNodeRegistry().then((res) => {
      setNodeRegistries(res)
    })
  }, [])

  return nodeRegistries
}
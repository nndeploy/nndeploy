import { FreeLayoutPluginContext } from "@flowgram.ai/free-layout-editor";

export function getNodeByName(name:string, clientContext: FreeLayoutPluginContext){
  const allNodes  = clientContext.document.getAllNodes()
  const find = allNodes.find(node=>{
    return node.form?.getValueIn('name_') == name
  })

  return find
}
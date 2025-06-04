import Mock from 'mockjs'

import file from './resource'
import {resourceHandler} from './pages/Resource'
import { nodeHandler} from './pages/Node'
//import nodeType from './nodeType'
import { workFlowHandler } from './pages/workFlow'

const mocks = [
  ...file, 
  ...resourceHandler, 
  ...workFlowHandler,
  ...nodeHandler, 
 // ...nodeType

]

// mock请求方法放在这里统一处理,1是简便写法,2是如果请求路径需要加统一前缀或域名,可以在这里处理
for (const i of mocks) {
  console.log('listen url', i.url)
  Mock.mock(i.url, i.type, i.response)
}
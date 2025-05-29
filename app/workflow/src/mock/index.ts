import Mock from 'mockjs'

import file from './resource'
import {resourceHandler} from './pages/Layout/Backend/Resource'
import node from './pages/Layout/Backend/Node'
import nodeType from './nodeType'

const mocks = [
  ...file, 
  ...resourceHandler, 
  ...node, 
  ...nodeType

]

// mock请求方法放在这里统一处理,1是简便写法,2是如果请求路径需要加统一前缀或域名,可以在这里处理
for (const i of mocks) {
  console.log('listen url', i.url)
  Mock.mock(i.url, i.type, i.response)
}
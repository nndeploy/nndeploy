var Mock = require('mockjs')
const Random = Mock.Random;
var data = Mock.mock({
    // 属性 list 的值是一个数组，其中含有 1 到 10 个元素
    'list|10-10': [{
        // 属性 id 是一个自增数，起始值为 1，每次增 1
        'id|+1': 1
    }]
})

Random.extend({
    constellation: function(date) {
        var constellations = ['白羊座', '金牛座', '双子座', '巨蟹座', '狮子座', '处女座', '天秤座', '天蝎座', '射手座', '摩羯座', '水瓶座', '双鱼座']
        return this.pick(constellations)
    }
})

// => "天蝎座"
var  constellation = Mock.mock({
    constellation: '@CONSTELLATION'
})

console.log(JSON.stringify(constellation, null, 2))


// 输出结果
// console.log(JSON.stringify(data, null, 4))

// console.log(JSON.stringify(data, null, 2))
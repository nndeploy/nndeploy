#ifndef _NNDEPLOY_DEVICE_UTIL_H_
#define _NNDEPLOY_DEVICE_UTIL_H_

namespace nndeploy {
namespace device {

template <typename T>
base::Status insertStream(int &index, std::map<int, T> &stream_map,
                          const T &stream) {
  if (stream_map.find(index) == stream_map.end()) {
    stream_map[index] = stream;
    return base::kStatusCodeOk;
  } else {
    if (index == INT_MAX - 1) {
      index = 0;
    }
    index++;
    return insertStream(index, stream_map, stream);
  }
}
template <typename T>
int updateStreamIndex(std::map<int, T> &stream_map) {
  // 得到map中倒数第二个元素
  auto iter = stream_map.end();
  iter--;
  iter--;
  int stream_index = iter->first;
  return stream_index;
}

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_DEVICE_UTIL_H_ */

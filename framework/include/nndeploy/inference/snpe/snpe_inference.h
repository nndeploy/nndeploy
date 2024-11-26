#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_convert.h"
#include "nndeploy/inference/snpe/snpe_include.h"
#include "nndeploy/inference/snpe/snpe_inference_param.h"

namespace nndeploy {
namespace inference {

class SnpeInference : public Inference {
 public:
  SnpeInference(base::InferenceType type);
  virtual ~SnpeInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap& shape_map);

  virtual base::Status run();

  virtual device::Tensor* getOutputTensorAfterRun(
      const std::string& name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  size_t calcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank,
                          size_t elementSize);
  size_t getResizableDim();
  void setResizableDim(size_t resizableDim);

  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

  typedef unsigned int GLuint;

  // Helper function to fill a single entry of the UserBufferMap with the given
  // user-backed buffer
  void createUserBuffer(
      zdl::DlSystem::UserBufferMap& userBufferMap,
      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
          snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE>& snpe, const char* name,
      const bool isTfNBuffer, int bitWidth);

  void createUserBuffer(
      zdl::DlSystem::UserBufferMap& userBufferMap,
      std::unordered_map<std::string, GLuint>& applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
          snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE>& snpe, const char* name);

  // Create a UserBufferMap of the SNPE network inputs
  void createInputBufferMap(
      zdl::DlSystem::UserBufferMap& inputMap,
      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
          snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE>& snpe, const bool isTfNBuffer,
      int bitWidth);

  void createInputBufferMap(
      zdl::DlSystem::UserBufferMap& inputMap,
      std::unordered_map<std::string, GLuint>& applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
          snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE>& snpe);

  // Create a UserBufferMap of the SNPE network outputs
  void createOutputBufferMap(
      zdl::DlSystem::UserBufferMap& outputMap,
      std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
          snpeUserBackedBuffers,
      std::unique_ptr<zdl::SNPE::SNPE>& snpe, const bool isTfNBuffer,
      int bitWidth);

 private:
  size_t resizable_dim;

  SnpeBuffer_Type_t buffer_type_;

  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  zdl::DlSystem::UserBufferMap input_map_;
  zdl::DlSystem::UserBufferMap output_map_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
      snpe_user_input_buffers_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
      snpe_user_output_buffers_;
  std::unordered_map<std::string, std::vector<uint8_t>>
      application_input_buffers_;
  std::unordered_map<std::string, std::vector<uint8_t>>
      application_output_buffers_;
  std::unique_ptr<zdl::DlSystem::ITensor> inputTensor_;
};

#define DEADBEAF_PTR (void*)(static_cast<intptr_t>(0xdeadbeaf))

static std::size_t getSizeByDim(const std::vector<size_t>& dim) {
  return std::accumulate(std::begin(dim), std::end(dim), 1,
                         std::multiplies<size_t>());
}

static void printArray(const size_t* start, const size_t* end) {
  (void)std::for_each(start, end,
                      [](const size_t i) { std::cout << i << " "; });
}

static bool isPassthrough(const std::string& type) {
  std::string lower;
  (void)std::transform(std::begin(type), std::end(type),
                       std::back_inserter(lower),
                       [](const char c) { return std::tolower(c); });
  if (lower != std::string("passthrough")) {
    std::cerr << "isPassthrough expecting type passthrough got " << type
              << std::endl;
    return false;
  }
  return true;
}

class UdlPassthrough final : public zdl::DlSystem::IUDL {
 public:
  UdlPassthrough(const UdlPassthrough&) = delete;
  UdlPassthrough& operator=(const UdlPassthrough&) = delete;

  /**
   * @brief UDLContext by value but it has move operation
   */
  UdlPassthrough(zdl::DlSystem::UDLContext context) : m_Context(context) {}

  /**
   * @brief Setup User's environment.
   *        This is being called by DnnRunTime framework
   *        to let the user opportunity to setup anything
   *        which is needed for running user defined layers
   * @return true on success, false otherwise
   */
  virtual bool setup(void* cookie, size_t insz, const size_t** indim,
                     const size_t* indimsz, size_t outsz, const size_t** outdim,
                     const size_t* outdimsz) override;

  /**
   * Close the instance. Invoked by DnnRunTime to let
   * the user the opportunity to close handels etc...
   */
  virtual void close(void* cookie) noexcept override;

  /**
   * Execute the user defined layer
   * will contain the return value/output tensor
   */
  virtual bool execute(void* cookie, const float** input,
                       float** output) override;

 private:
  zdl::DlSystem::UDLContext m_Context;
  // this is a*b*c*...*n
  std::vector<size_t> m_OutSzDim;
  // cache the insz/outsz of the incoming
  size_t m_Insz = 0;
  // No need for this since in passthrough its all the same
  // size_t m_Outsz = 0;
};

static zdl::DlSystem::IUDL* MyUDLFactory(void* cookie,
                                         const zdl::DlSystem::UDLContext* c) {
  std::cout << "In MyUDLFactory!" << std::endl;
  if (!c) return nullptr;
  if (cookie != DEADBEAF_PTR) {
    std::cerr << "MyUDLFactory cookie should be 0xdeadbeaf" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!isPassthrough(c->getType())) {
    std::cerr << "MyUDLFactory expecting Passthrough layer, got "
              << c->getType() << std::endl;
    return nullptr;
  }
  return new UdlPassthrough(*c);
}

bool UdlPassthrough::setup(void* cookie, size_t insz, const size_t* indim[],
                           const size_t indimsz[], size_t outsz,
                           const size_t* outdim[], const size_t outdimsz[]) {
  if (cookie != DEADBEAF_PTR) {
    std::cerr << "UdlPassthrough::setup() cookie should be 0xdeadbeaf"
              << std::endl;
    return false;
  }

  // FIXME we need to use proper logging here not using streams
  std::cout << "UdlPassthrough::setup() of name " << m_Context.getName()
            << " and of type " << m_Context.getType() << std::endl;

  if (!isPassthrough(m_Context.getType())) {
    std::cerr << "UdlPassthrough::setup() expecting passthrough layer type got "
              << m_Context.getType() << std::endl;
    return false;
  }

  std::cout << "                        input array size " << insz << std::endl;
  std::cout << "                        output array size " << outsz
            << std::endl;

  // print the input/output dims
  std::cout << "UdlPassthrough::setup() input dims\n";
  (void)std::copy(indimsz, indimsz + insz,
                  std::ostream_iterator<size_t>(std::cout, " "));
  std::cout << std::endl;
  size_t idx = 0;
  (void)std::for_each(indim, indim + insz, [&idx, &indimsz](const size_t* arr) {
    std::cout << "[";
    printArray(arr, arr + indimsz[idx]);
    std::cout << "]";
    ++idx;
  });
  std::cout << std::endl;
  std::cout << "UdlPassthrough::setup() output dims\n";
  (void)std::copy(outdimsz, outdimsz + insz,
                  std::ostream_iterator<size_t>(std::cout, " "));
  std::cout << std::endl;
  idx = 0;
  (void)std::for_each(outdim, outdim + outsz,
                      [&idx, &outdimsz](const size_t* arr) {
                        std::cout << "[";
                        printArray(arr, arr + outdimsz[idx]);
                        std::cout << "]";
                        ++idx;
                      });
  std::cout << std::endl;

  if (insz != outsz) {
    std::cerr << "UdlPassthrough::setup() not the same number of dim, in:"
              << insz << " != : " << outsz << std::endl;
    return false;
  }
  m_Insz = insz;
  size_t cnt = insz;

  // If the user want to refer to the indim[] and outdim[],
  // he/she needs to make a copy of this arrays.
  // After setup, these arrays are destroyes, so you cannot cache it as is
  m_OutSzDim.reserve(cnt);
  while (cnt-- > 0) {
    // compute dims and compare. keep the output dim
    const size_t* indims = indim[cnt];
    const size_t inszdim =
        getSizeByDim(std::vector<size_t>(indims, indims + indimsz[cnt]));
    const size_t* outdims = outdim[cnt];  // insz == outsz
    m_OutSzDim[cnt] =
        getSizeByDim(std::vector<size_t>(outdims, outdims + outdimsz[cnt]));

    std::cout << "UdlPassthrough::setup() input size for index " << cnt
              << " is dim: " << inszdim << ", output: " << m_OutSzDim[cnt]
              << std::endl;
    if (inszdim != m_OutSzDim[cnt]) {
      std::cerr << "UdlPassthrough::setup() not the same overall dim, in:"
                << inszdim << " != out: " << m_OutSzDim[cnt] << std::endl;
      return false;
    }
  }
  // parse the Passthrough params
  const uint8_t* blob = m_Context.getBlob();
  std::cout << "UdlPassthrough::setup() got blob size " << m_Context.getSize()
            << std::endl;
  if (!blob) {
    std::cout << "UdlPassthrough::setup() got null blob " << std::endl;
    return false;
  }
  // Python packing is this way:
  // self._blob = struct.pack('I', params.blob_count)
  // 'I' here means 32bit - https://docs.python.org/2/library/struct.html
  std::cout << "UdlPassthrough::setup() got blob content "
            << *(reinterpret_cast<const int32_t*>(blob)) << std::endl;
  return true;
}

void UdlPassthrough::close(void* cookie) noexcept {
  if (cookie != DEADBEAF_PTR) {
    std::cerr << "UdlPassthrough::close() cookie should be 0xdeadbeaf"
              << std::endl;
  }
  std::cout << "UdlPassthrough::close()" << std::endl;
  delete this;
}

bool UdlPassthrough::execute(void* cookie, const float** input,
                             float** output) {
  if (cookie != DEADBEAF_PTR) {
    std::cerr << "UdlPassthrough::execute() cookie should be 0xdeadbeaf"
              << std::endl;
    return false;
  }
  std::cout << "UdlPassthrough::execute() number of I/Os is:" << m_Insz
            << std::endl;
  // 0...m_OutSzDim --> going backwards
  // m_OutSzDim is assumed to be != 0
  size_t cnt = m_Insz;
  while (cnt-- > 0) {
    std::cout << std::dec;
    std::cout << "UdlPassthrough::execute() running index " << cnt << std::endl;
    size_t dim = sizeof(float) * m_OutSzDim[cnt];
    std::cout << "UdlPassthrough::execute() dims (a*b*c*...) is:"
              << m_OutSzDim[cnt] << std::endl;
    std::cout << "UdlPassthrough::execute() dim(total number of bytes) is:"
              << dim << std::endl;

    const float* i = input[cnt];
    float* o = output[cnt];
    if (!i || !o) {
      std::cerr << "Input or output cannot be 0" << std::endl;
      return false;
    }
    std::cout << "input: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << i << std::endl;
    std::cout << "output: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << o << std::endl;
    std::memcpy(o, i, dim);
  }
  return true;
}

}  // namespace inference
}  // namespace nndeploy

#endif
#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INCLUDE_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INCLUDE_H_

#include <DlSystem/ITensor.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <DlSystem/String.hpp>
#include <DlSystem/StringList.hpp>
#include <DlSystem/TensorMap.hpp>
#include <DlSystem/TensorShape.hpp>
#include <DlSystem/UserBufferMap.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEFactory.hpp>

#include "DiagLog/IDiagLog.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/IUDL.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/UDLContext.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "SNPE/SNPEBuilder.hpp"


namespace nndeploy {
namespace inference {

enum SnpeRuntime_t : int {
  SNPE_RT_CPU = 0x0000,
  SNPE_RT_GPU,
  SNPE_RT_DSP,
  SNPE_RT_AIP
};

enum SnpePerfMode_t : int {
  SNPE_PERF_BALANCED = 0x0000,
  SNPE_PERF_HIGH_PERFORMANCE,
  SNPE_PERF_POWER_SAVER,
  SNPE_PERF_SYSTEM_SETTINGS,
  SNPE_PERF_SUSTAINED_HIGH_PERFORMANCE,
  SNPE_PERF_BURST
};

enum SnpeProfilingLevel_t : int {
  SNPE_PROFILING_OFF = 0x0000,
  SNPE_PROFILING_BASIC,
  SNPE_PROFILING_DETAILED
};

enum SnpeBuffer_Type_t : int {
  USERBUFFER_FLOAT = 0x0000,
  USERBUFFER_TF8,
  ITENSOR,
  USERBUFFER_TF16
};

enum SnpeUserBufferSource_Type_t : int { CPUBUFFER = 0x0000, GLBUFFER };

}  // namespace inference
}  // namespace nndeploy

#endif
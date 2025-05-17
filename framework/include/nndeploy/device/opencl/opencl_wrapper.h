// Tencent is pleased to support the open source community by making TNN
// available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_WRAPPER_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_WRAPPER_H_

#ifdef WIN32
#define NOMINMAX
#include <windows.h>
// Do not remove following statement.
// windows.h replace LoadLibrary with LoadLibraryA, which cause compiling issue
// of TNN.
#undef LoadLibrary
#endif

#include <memory>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/opencl/opencl_include.h"

// support opencl min version is 1.1
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#include <CL/cl_egl.h>

#define CHECK_NOTNULL(X)                   \
  NNDEPLOY_ASSERT(X != NULL)               \
  if (X == NULL) {                         \
    NNDEPLOY_LOGE("OpenCL API is null\n"); \
  }

#define CHECK_CL_SUCCESS(error)                             \
  if (error != CL_SUCCESS) {                                \
    NNDEPLOY_LOGE("OpenCL ERROR CODE : %d \n", (int)error); \
  }

#define CHECK_NNDEPLOY_OK(error)                        \
  if (error != NNDEPLOY_OK) {                           \
    NNDEPLOY_LOGE("%s\n", error.description().c_str()); \
    return error;                                       \
  }

#ifdef NNDEPLOY_USE_OPENCL_WRAPPER

namespace nndeploy {
namespace device {

// OpenCLSymbols is a opencl function wrapper.
// if device not support opencl or opencl target function,
// app will not crash and can get error code.
class OpenCLSymbols {
 public:
  static OpenCLSymbols *GetInstance();

  ~OpenCLSymbols();
  OpenCLSymbols(const OpenCLSymbols &) = delete;
  OpenCLSymbols &operator=(const OpenCLSymbols &) = delete;

  bool LoadOpenCLLibrary();
  bool UnLoadOpenCLLibrary();
  // get platfrom id
  using clGetPlatformIDsFunc = cl_int(CL_API_CALL *)(cl_uint, cl_platform_id *,
                                                     cl_uint *);
  // get platform info
  using clGetPlatformInfoFunc = cl_int(CL_API_CALL *)(cl_platform_id,
                                                      cl_platform_info, size_t,
                                                      void *, size_t *);
  // build program
  using clBuildProgramFunc = cl_int(CL_API_CALL *)(
      cl_program, cl_uint, const cl_device_id *, const char *,
      void(CL_CALLBACK *pfn_notify)(cl_program, void *), void *);
  // enqueue run kernel
  using clEnqueueNDRangeKernelFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
      const size_t *, cl_uint, const cl_event *, cl_event *);
  // set kernel parameter
  using clSetKernelArgFunc = cl_int(CL_API_CALL *)(cl_kernel, cl_uint, size_t,
                                                   const void *);
  using clRetainMemObjectFunc = cl_int(CL_API_CALL *)(cl_mem);
  using clReleaseMemObjectFunc = cl_int(CL_API_CALL *)(cl_mem);
  using clEnqueueUnmapMemObjectFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  using clRetainCommandQueueFunc =
      cl_int(CL_API_CALL *)(cl_command_queue command_queue);
  // create context
  using clCreateContextFunc = cl_context(CL_API_CALL *)(
      const cl_context_properties *, cl_uint, const cl_device_id *,
      void(CL_CALLBACK *)(  // NOLINT(readability/casting)
          const char *, const void *, size_t, void *),
      void *, cl_int *);
  using clEnqueueCopyImageFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *,
      const size_t *, cl_uint, const cl_event *, cl_event *);

  using clCreateContextFromTypeFunc = cl_context(CL_API_CALL *)(
      const cl_context_properties *, cl_device_type,
      void(CL_CALLBACK *)(  // NOLINT(readability/casting)
          const char *, const void *, size_t, void *),
      void *, cl_int *);
  using clReleaseContextFunc = cl_int(CL_API_CALL *)(cl_context);
  using clWaitForEventsFunc = cl_int(CL_API_CALL *)(cl_uint, const cl_event *);
  using clReleaseEventFunc = cl_int(CL_API_CALL *)(cl_event);
  using clEnqueueWriteBufferFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint,
      const cl_event *, cl_event *);
  using clEnqueueReadBufferFunc = cl_int(CL_API_CALL *)(cl_command_queue,
                                                        cl_mem, cl_bool, size_t,
                                                        size_t, void *, cl_uint,
                                                        const cl_event *,
                                                        cl_event *);
  using clGetProgramBuildInfoFunc = cl_int(CL_API_CALL *)(cl_program,
                                                          cl_device_id,
                                                          cl_program_build_info,
                                                          size_t, void *,
                                                          size_t *);
  using clRetainProgramFunc = cl_int(CL_API_CALL *)(cl_program program);
  using clEnqueueMapBufferFunc = void *(CL_API_CALL *)(cl_command_queue, cl_mem,
                                                       cl_bool, cl_map_flags,
                                                       size_t, size_t, cl_uint,
                                                       const cl_event *,
                                                       cl_event *, cl_int *);
  using clEnqueueMapImageFunc =
      void *(CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, cl_map_flags,
                            const size_t *, const size_t *, size_t *, size_t *,
                            cl_uint, const cl_event *, cl_event *, cl_int *);
  using clCreateCommandQueueFunc = cl_command_queue(CL_API_CALL *)(  // NOLINT
      cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
  using clGetCommandQueueInfoFunc = cl_int(CL_API_CALL *)(cl_command_queue,
                                                          cl_command_queue_info,
                                                          size_t, void *,
                                                          size_t *);
  using clReleaseCommandQueueFunc = cl_int(CL_API_CALL *)(cl_command_queue);
  using clCreateProgramWithBinaryFunc = cl_program(CL_API_CALL *)(
      cl_context, cl_uint, const cl_device_id *, const size_t *,
      const unsigned char **, cl_int *, cl_int *);
  using clRetainContextFunc = cl_int(CL_API_CALL *)(cl_context context);
  using clGetContextInfoFunc = cl_int(CL_API_CALL *)(cl_context,
                                                     cl_context_info, size_t,
                                                     void *, size_t *);
  using clReleaseProgramFunc = cl_int(CL_API_CALL *)(cl_program program);
  // flush command queue to target device
  using clFlushFunc = cl_int(CL_API_CALL *)(cl_command_queue command_queue);
  // sync device command queue
  using clFinishFunc = cl_int(CL_API_CALL *)(cl_command_queue command_queue);
  using clGetProgramInfoFunc = cl_int(CL_API_CALL *)(cl_program,
                                                     cl_program_info, size_t,
                                                     void *, size_t *);
  // create kernel with kernel name
  using clCreateKernelFunc = cl_kernel(CL_API_CALL *)(cl_program, const char *,
                                                      cl_int *);
  using clRetainKernelFunc = cl_int(CL_API_CALL *)(cl_kernel kernel);
  using clCreateBufferFunc = cl_mem(CL_API_CALL *)(cl_context, cl_mem_flags,
                                                   size_t, void *, cl_int *);
  // create image 2d
  using clCreateImage2DFunc = cl_mem(CL_API_CALL *)(cl_context,  // NOLINT
                                                    cl_mem_flags,
                                                    const cl_image_format *,
                                                    size_t, size_t, size_t,
                                                    void *, cl_int *);
  // create image 3d
  using clCreateImage3DFunc = cl_mem(CL_API_CALL *)(cl_context,  // NOLINT
                                                    cl_mem_flags,
                                                    const cl_image_format *,
                                                    size_t, size_t, size_t,
                                                    size_t, size_t, void *,
                                                    cl_int *);
  // crete program with source code
  using clCreateProgramWithSourceFunc = cl_program(CL_API_CALL *)(
      cl_context, cl_uint, const char **, const size_t *, cl_int *);
  using clReleaseKernelFunc = cl_int(CL_API_CALL *)(cl_kernel kernel);
  using clGetDeviceInfoFunc = cl_int(CL_API_CALL *)(cl_device_id,
                                                    cl_device_info, size_t,
                                                    void *, size_t *);
  // get device id, two device have different device id
  using clGetDeviceIDsFunc = cl_int(CL_API_CALL *)(cl_platform_id,
                                                   cl_device_type, cl_uint,
                                                   cl_device_id *, cl_uint *);
  using clRetainEventFunc = cl_int(CL_API_CALL *)(cl_event);
  using clGetKernelWorkGroupInfoFunc =
      cl_int(CL_API_CALL *)(cl_kernel, cl_device_id, cl_kernel_work_group_info,
                            size_t, void *, size_t *);
  using clGetEventInfoFunc = cl_int(CL_API_CALL *)(
      cl_event event, cl_event_info param_name, size_t param_value_size,
      void *param_value, size_t *param_value_size_ret);
  using clGetEventProfilingInfoFunc = cl_int(CL_API_CALL *)(
      cl_event event, cl_profiling_info param_name, size_t param_value_size,
      void *param_value, size_t *param_value_size_ret);
  using clGetImageInfoFunc = cl_int(CL_API_CALL *)(cl_mem, cl_image_info,
                                                   size_t, void *, size_t *);
  using clEnqueueCopyBufferToImageFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *,
      cl_uint, const cl_event *, cl_event *);
  using clEnqueueCopyImageToBufferFunc = cl_int(CL_API_CALL *)(
      cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t,
      cl_uint, const cl_event *, cl_event *);
  using clCreateFromEGLImageKHRFunc = cl_mem(CL_API_CALL *)(
      cl_context, CLeglDisplayKHR, CLeglImageKHR, cl_mem_flags,
      const cl_egl_image_properties_khr *, cl_int *);
  using clEnqueueAcquireEGLObjectsKHRFunc = cl_int(CL_API_CALL *)(
      cl_command_queue /* command_queue */, cl_uint /* num_objects */,
      const cl_mem * /* mem_objects */, cl_uint /* num_events_in_wait_list */,
      const cl_event * /* event_wait_list */, cl_event * /* event */);
  using clEnqueueReleaseEGLObjectsKHRFunc = cl_int(CL_API_CALL *)(
      cl_command_queue /* command_queue */, cl_uint /* num_objects */,
      const cl_mem * /* mem_objects */, cl_uint /* num_events_in_wait_list */,
      const cl_event * /* event_wait_list */, cl_event * /* event */);
  using clEnqueueReadImageFunc = cl_int(CL_API_CALL *)(
      cl_command_queue /* command_queue */, cl_mem /* image */,
      cl_bool /* blocking_read */, const size_t * /* origin[3] */,
      const size_t * /* region[3] */, size_t /* row_pitch */,
      size_t /* slice_pitch */, void * /* ptr */,
      cl_uint /* num_events_in_wait_list */,
      const cl_event * /* event_wait_list */, cl_event * /* event */);
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
  using clRetainDeviceFunc = cl_int(CL_API_CALL *)(cl_device_id);
  using clReleaseDeviceFunc = cl_int(CL_API_CALL *)(cl_device_id);
  using clCreateImageFunc = cl_mem(CL_API_CALL *)(cl_context, cl_mem_flags,
                                                  const cl_image_format *,
                                                  const cl_image_desc *, void *,
                                                  cl_int *);
  using clEnqueueAcquireGLObjectsFunc = cl_int (*)(
      cl_command_queue queue /* command_queue */,
      cl_uint n_obj /* num_objects */, const cl_mem *mem /* mem_objects */,
      cl_uint n_elist /* num_events_in_wait_list */,
      const cl_event *wlist /* event_wait_list */, cl_event *event /* event */);
  using clEnqueueReleaseGLObjectsFunc = cl_int (*)(
      cl_command_queue queue /* command_queue */,
      cl_uint n_obj /* num_objects */, const cl_mem *mem /* mem_objects */,
      cl_uint n_elist /* num_events_in_wait_list */,
      const cl_event *wlist /* event_wait_list */, cl_event *event /* event */);
  using clCreateFromGLTextureFunc = cl_mem (*)(cl_context context /* context */,
                                               cl_mem_flags flags /* flags */,
                                               cl_GLenum target /* target */,
                                               cl_GLint level /* miplevel */,
                                               cl_GLuint tex /* texture */,
                                               cl_int *err /* errcode_ret */);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  // opencl 2.0 can get sub group info and wave size.
  using clGetKernelSubGroupInfoKHRFunc =
      cl_int(CL_API_CALL *)(cl_kernel, cl_device_id, cl_kernel_sub_group_info,
                            size_t, const void *, size_t, void *, size_t *);
  using clCreateCommandQueueWithPropertiesFunc =
      cl_command_queue(CL_API_CALL *)(cl_context, cl_device_id,
                                      const cl_queue_properties *, cl_int *);
  using clGetExtensionFunctionAddressFunc = void *(CL_API_CALL *)(const char *);
#endif

#define NNDEPLOY_CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr

  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clBuildProgram);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clSetKernelArg);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseKernel);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateBuffer);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateImage2D);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateImage3D);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainKernel);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateKernel);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetProgramInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clFlush);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clFinish);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseProgram);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainContext);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetContextInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateCommandQueue);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetCommandQueueInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueMapImage);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueCopyImage);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainProgram);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clWaitForEvents);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseEvent);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateContext);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateContextFromType);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseContext);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainCommandQueue);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainMemObject);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseMemObject);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainEvent);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetEventInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetImageInfo);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueCopyBufferToImage);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueCopyImageToBuffer);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateFromEGLImageKHR);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueAcquireEGLObjectsKHR);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueReleaseEGLObjectsKHR);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueReadImage);
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clRetainDevice);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clReleaseDevice);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateImage);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueAcquireGLObjects);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clEnqueueReleaseGLObjects);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateFromGLTexture);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetKernelSubGroupInfoKHR);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
  NNDEPLOY_CL_DEFINE_FUNC_PTR(clGetExtensionFunctionAddress);
#endif

#undef NNDEPLOY_CL_DEFINE_FUNC_PTR

 private:
  OpenCLSymbols();
  bool LoadLibraryFromPath(const std::string &path);

 private:
  static std::shared_ptr<OpenCLSymbols> opencl_symbols_singleton_;
#ifdef WIN32
  HMODULE handle_ = nullptr;
#else
  void *handle_ = nullptr;
#endif
};

}  // namespace device
}  // namespace nndeploy
#endif  // NNDEPLOY_USE_OPENCL_WRAPPER
#endif  // _NNDEPLOY_DEVICE_OPENCL_OPENCL_WRAPPER_H_
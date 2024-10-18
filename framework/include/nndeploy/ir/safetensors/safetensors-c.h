// SPDX-License-Identifier: MIT
// Copyright 2023 - Present, Syoyo Fujita.
// C binding to safetensors.hh
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAFETENSORS_C_MAX_DIM (8)

typedef enum safetensors_c_dtype {
  SAFETENSORS_C_DTYPE_INVALID,
  SAFETENSORS_C_BOOL,
  SAFETENSORS_C_UINT8,
  SAFETENSORS_C_INT8,
  SAFETENSORS_C_UINT16,
  SAFETENSORS_C_INT16,
  SAFETENSORS_C_FLOAT16,
  SAFETENSORS_C_BFLOAT16,
  SAFETENSORS_C_UINT32,
  SAFETENSORS_C_INT32,
  SAFETENSORS_C_FLOAT32,
  SAFETENSORS_C_FLOAT64,
  SAFETENSORS_C_UINT64,
  SAFETENSORS_C_INT64,
} safetensors_c_dtype_t;

// TODO: Use error code used in official rust implementation
typedef enum safetensors_c_status {
  SAFETENSORS_C_SUCCESS = 0,
  SAFETENSORS_C_FILE_NOT_FOUND = -1,
  SAFETENSORS_C_FILE_READ_FAILURE = -2,
  SAFETENSORS_C_FILE_WRITE_FAILURE = -3,
  SAFETENSORS_C_CORRUPTED_DATA = -4,
  SAFETENSORS_C_INVALID_SAFETENSORS = -5,
  SAFETENSORS_C_MALLOC_ERROR = -6,
  SAFETENSORS_C_INVALID_ARGUMENT = -7,
  SAFETENSORS_C_KEY_NOT_FOUND = -8
} safetensors_c_status_t;

typedef struct safetensors_c_safetensors {
  // opaque pointer to satetensors::safetensors_t
  void *ptr;
} safetensors_c_safetensors_t;

typedef struct safetensors_c_tensor {
  safetensors_c_dtype_t dtype;
  size_t shape[SAFETENSORS_C_MAX_DIM];
  uint32_t ndim;  // up to SAFETENSORS_C_MAX_DIM;

  size_t data_offsets[2];  // [BEGIN, END]

} safetensors_c_tensor_t;

void safetensors_c_init(safetensors_c_safetensors_t *st);
void safetensors_c_free(safetensors_c_safetensors_t *st);

void safetensors_c_tensor_init(safetensors_c_tensor_t *t);

//
// Currently there is no dynamically allocated members in tensor, so this API does nothing.
//
void safetensors_c_tensor_free(safetensors_c_tensor_t *t);

//
// Load safetensors from a file.
//
// @param[in] filename Filename(Assume UTF-8)
// @param[out] warn Warning message when failed. Must free the pointer after
// using it. Pass NULL if you don't need it.
// @param[out] err Error message when failed. Must free the pointer after using
// it. Pass NULL if you don't need it.
// @return `SAFETENSORS_C_SUCCESS` upon success.
//
safetensors_c_status_t safetensors_c_load_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char **warn,
    char **err);

//
// Load safetensors from a memory
//
// @param[in] addr Buffer address
// @param[in] nbytes Buffer size
// @param[in] filename Filename(Assume UTF-8). Optional. Can be NULL.
// @param[out] warn Warning message when failed. Must free the pointer after
// using it. Pass NULL if you don't need it.
// @param[out] err Error message when failed. Must free the pointer after using
// it. Pass NULL if you don't need it.
// @return `SAFETENSORS_C_SUCCESS` upon success.
//
safetensors_c_status_t safetensors_c_load_from_memory(
    const void *addr, const size_t bytes, const char *filename,
    safetensors_c_safetensors_t *st, char **warn, char **err);

// mmap version
// Still need to call `safetensors_c_safetensors_free` API to free JSON data in
// safetensors struct.
//
safetensors_c_status_t safetensors_c_mmap_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char **warn,
    char **err);
safetensors_c_status_t safetensors_c_mmap_from_memory(
    const char *filename, safetensors_c_safetensors_t *st, char **warn,
    char **err);

//
// Fee memory of safetensors struct.
// No need to call this API for mmaped safetensors data.
// @return 1 upon success, 0 failed.
//
int safetensors_c_safetensors_free(safetensors_c_safetensors_t *st);

///
/// @return The number of tensors.
uint32_t safetensors_c_num_tensors(const safetensors_c_safetensors_t *st);

///
/// Check if tensor item with `key` exists.
/// @return SAFETENSORS_C_SUCCESS upon success and fill `has_tensor`, others are error.
///
int safetensors_c_has_tensor(const safetensors_c_safetensors_t *st,
                               const char *key, int *has_tensor);

///
/// Get tensor item for `key`. Return NULL when no corresponding tensor item
/// exists for `key`.
/// Memory is **not** allocated for returned tensor value. No need for freeing `tensor` after using it.
///
/// @return SAFETENSORS_C_SUCCESS upon success and fill `tensor`, others are error.
int safetensors_c_get_tensor(const safetensors_c_safetensors_t *st,
                                     const char *key,
                                     safetensors_c_tensor_t *tensor);

///
/// Get tensor item at specified index. Return NULL when index is out-of-range.
/// exists for `key`.
/// Memory is **not** allocated for returned tensor value. No need for freeing `tensor` after using it.
///
/// @return SAFETENSORS_C_SUCCESS upon success and fill `tensor`, others are error.
int safetensors_c_get_tensor_at(const safetensors_c_safetensors_t *st,
                                       uint32_t index,
                                       char **value);

///
/// @return The number of metadata items.
///
uint32_t safetensors_c_num_metadata(const safetensors_c_safetensors_t *st);

///
/// Check if metadata item with `key` exists.
/// @return 1 if the metadata has item with `key`. 0 not.
int safetensors_c_has_metadata(const safetensors_c_safetensors_t *st,
                               const char *key);

///
/// Get metadata item for `key`. Return NULL when no corresponding metadata item
/// exists for `key`.
/// Memory will be allocated for `value`(even in mmap mode). Need to free `value` after using it.
///
const char *safetensors_c_get_metadata(const safetensors_c_safetensors_t *st,
                                       const char *key,
                                       char **value);

///
/// Get metadata item at specified index. Return NULL when index is out-of-range.
/// exists for `key`.
/// Memory will be allocated for `value`(even in mmap mode). Need to free `value` after using it.
///
const char *safetensors_c_get_metadata_at(const safetensors_c_safetensors_t *st,
                                       uint32_t index,
                                       char **value);

// TODO: Write API

#ifdef __cplusplus
}  // extern "C"
#endif

#if defined(SAFETENSORS_C_IMPLEMENTATION)

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#if !defined(SAFETENSORS_CPP_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#endif
#include "safetensors.hh"

namespace safetensors_c {
namespace detail {

safetensors_c_dtype_t convert_cpp_dtype(safetensors::dtype dtype) {
  switch (dtype) {
    case safetensors::dtype::kBOOL: return SAFETENSORS_C_BOOL;
    case safetensors::dtype::kUINT8: return SAFETENSORS_C_UINT8;
    case safetensors::dtype::kINT8: return SAFETENSORS_C_INT8;
    case safetensors::dtype::kUINT16: return SAFETENSORS_C_UINT16;
    case safetensors::dtype::kINT16: return SAFETENSORS_C_INT16;
    case safetensors::dtype::kFLOAT16: return SAFETENSORS_C_FLOAT16;
    case safetensors::dtype::kBFLOAT16: return SAFETENSORS_C_BFLOAT16;
    case safetensors::dtype::kUINT32: return SAFETENSORS_C_UINT32;
    case safetensors::dtype::kINT32: return SAFETENSORS_C_INT32;
    case safetensors::dtype::kFLOAT32: return SAFETENSORS_C_FLOAT32;
    case safetensors::dtype::kFLOAT64: return SAFETENSORS_C_FLOAT64;
    case safetensors::dtype::kUINT64: return SAFETENSORS_C_UINT64;
    case safetensors::dtype::kINT64: return SAFETENSORS_C_INT64;
  }

  return SAFETENSORS_C_DTYPE_INVALID;
}

}
};

#ifdef __cplusplus
extern "C" {
#endif

void safetensors_c_init(safetensors_c_safetensors_t *st) {
  if (!st) {
    return;
  }
  st->ptr = nullptr;
}

int safetensors_c_is_mmaped(
  const safetensors_c_safetensors_t *st, int *is_mmaped) {

  if (!st || !is_mmaped) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  if (!st->ptr) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  safetensors::safetensors_t *cppst = reinterpret_cast<safetensors::safetensors_t *>(st->ptr);

  (*is_mmaped) = cppst->mmaped ? 1 : 0;

  return SAFETENSORS_C_SUCCESS;

}

int safetensors_c_get_databuffer(
  const safetensors_c_safetensors_t *st,
  const void **addr, size_t *nbytes) {

  if (!st || !addr || !nbytes) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  if (!st->ptr) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  safetensors::safetensors_t *cppst = reinterpret_cast<safetensors::safetensors_t *>(st->ptr);

  if (cppst->mmaped) {
    (*addr) = cppst->mmap_addr;
    (*nbytes) = cppst->mmap_size;
  } else {
    (*addr) = reinterpret_cast<const void *>(cppst->storage.data());
    (*nbytes) = cppst->storage.size();
  }

  return SAFETENSORS_C_SUCCESS;

}

int safetensors_c_has_tensor(const safetensors_c_safetensors_t *st,
  const char *key,
  int *has_tensor) {

  if (!st || !key || !has_tensor) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  if (!st->ptr) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  safetensors::safetensors_t *cppst = reinterpret_cast<safetensors::safetensors_t *>(st->ptr);

  (*has_tensor) = cppst->tensors.count(key) ? 1 : 0;

  return SAFETENSORS_C_SUCCESS;
}

int safetensors_c_get_tensor(const safetensors_c_safetensors_t *st,
  const char *key,
  safetensors_c_tensor_t *tensor_out) {

  if (!st || !key || !tensor_out) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  if (!st->ptr) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  safetensors::safetensors_t *cppst = reinterpret_cast<safetensors::safetensors_t *>(st->ptr);

  if (!cppst->tensors.count(key)) {
    return SAFETENSORS_C_KEY_NOT_FOUND;
  }

  safetensors::tensor_t ts;

  if (!cppst->tensors.at(key, &ts)) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  if (ts.shape.size() >= SAFETENSORS_C_MAX_DIM) {
    return SAFETENSORS_C_CORRUPTED_DATA;
  }

  tensor_out->dtype = safetensors_c::detail::convert_cpp_dtype(ts.dtype);
  tensor_out->ndim = ts.shape.size();
  for (size_t i = 0; i < ts.shape.size(); i++) {
    tensor_out->shape[i] = ts.shape[i];
  }
  tensor_out->data_offsets[0] = ts.data_offsets[0];
  tensor_out->data_offsets[1] = ts.data_offsets[1];

  return SAFETENSORS_C_SUCCESS;
}

safetensors_c_status_t safetensors_c_load_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char **warn,
    char **err) {

  if (!st) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  safetensors_c_init(st);

  std::string _warn;
  std::string _err;

  // TODO: First check if file exists

  safetensors::safetensors_t *cpp_st = new safetensors::safetensors_t();
  if (!safetensors::load_from_file(filename, cpp_st, &_warn, &_err)) {

    delete cpp_st;

    if (_err.size()) {
      char *err_msg = reinterpret_cast<char *>(malloc(_err.size()));
      if (!err_msg) {
        return SAFETENSORS_C_MALLOC_ERROR;
      }

      (*err) = err_msg;
    }

    return SAFETENSORS_C_FILE_READ_FAILURE;
  }

  st->ptr = cpp_st;

  return SAFETENSORS_C_SUCCESS;
}

safetensors_c_status_t safetensors_c_load_from_memory(
    const void *addr, const size_t nbytes, const char *filename,
    safetensors_c_safetensors_t *st, char **warn, char **err) {

  if (!st) {
    return SAFETENSORS_C_INVALID_ARGUMENT;
  }

  safetensors_c_init(st);

  std::string _warn;
  std::string _err;

  // TODO: First check if file exists

  safetensors::safetensors_t *cpp_st = new safetensors::safetensors_t();
  if (!safetensors::load_from_memory(reinterpret_cast<const uint8_t *>(addr), nbytes, filename, cpp_st, &_warn, &_err)) {

    delete cpp_st;

    if (_err.size()) {
      char *err_msg = reinterpret_cast<char *>(malloc(_err.size()));
      if (!err_msg) {
        return SAFETENSORS_C_MALLOC_ERROR;
      }

      (*err) = err_msg;
    }

    return SAFETENSORS_C_FILE_READ_FAILURE;
  }

  st->ptr = cpp_st;

  return SAFETENSORS_C_SUCCESS;
}

void safetensors_c_free(safetensors_c_safetensors_t *st) {
  if (!st) {
    return;
  }

  safetensors::safetensors_t *p =
      reinterpret_cast<safetensors::safetensors_t *>(st->ptr);

  delete p;

  st->ptr = nullptr;

  return;
}

void safetensors_c_tensor_init(safetensors_c_tensor_t *st) {
  if (!st) {
    return;
  }

  st->dtype = SAFETENSORS_C_DTYPE_INVALID;
  memset(st->shape, 0, sizeof(size_t) * SAFETENSORS_C_MAX_DIM);
  st->ndim = 0;
  st->data_offsets[0] = 0;
  st->data_offsets[1] = 0;
}

void safetensors_c_tensor_free(safetensors_c_tensor_t *st) {
  if (!st) {
    return;
  }

  // No members to free for tensor_t
  return;
}


#ifdef __cplusplus
}  // extern "C"
#endif

#endif

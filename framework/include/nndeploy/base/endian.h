/*!
 * \file endian.h
 * \brief ref dmlc-core/include/dmlc/endian.h
 */

#ifndef _NNDEPLOY_BASE_ENDIAN_H_
#define _NNDEPLOY_BASE_ENDIAN_H_

#ifdef NNDEPLOY_CMAKE_LITTLE_ENDIAN
  // If compiled with CMake, use CMake's endian detection logic
  #define NNDEPLOY_LITTLE_ENDIAN NNDEPLOY_CMAKE_LITTLE_ENDIAN
#else
  #if defined(__APPLE__) || defined(_WIN32)
    #define NNDEPLOY_LITTLE_ENDIAN 1
  #elif defined(__GLIBC__) || defined(__GNU_LIBRARY__) \
        || defined(__ANDROID__) || defined(__RISCV__)
    #include <endian.h>
    #define NNDEPLOY_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
  #elif defined(__FreeBSD__) || defined(__OpenBSD__)
    #include <sys/endian.h>
    #define NNDEPLOY_LITTLE_ENDIAN (_BYTE_ORDER == _LITTLE_ENDIAN)
  #elif defined(__QNX__)
    #include <sys/param.h>
    #define NNDEPLOY_LITTLE_ENDIAN (BYTE_ORDER == LITTLE_ENDIAN)
  #elif defined(__EMSCRIPTEN__) || defined(__hexagon__)
    #define NNDEPLOY_LITTLE_ENDIAN 1
  #elif defined(__sun) || defined(sun)
    #include <sys/isa_defs.h>
    #if defined(_LITTLE_ENDIAN)
      #define NNDEPLOY_LITTLE_ENDIAN 1
    #else
      #define NNDEPLOY_LITTLE_ENDIAN 0
    #endif
  #else
    #error "Unable to determine endianness of your machine; use CMake to compile"
  #endif
#endif

/*!
 * \brief Use little endian for binary serialization
 *  if this is set to 0, use big endian.
 */
#ifndef NNDEPLOY_IO_USE_LITTLE_ENDIAN
#define NNDEPLOY_IO_USE_LITTLE_ENDIAN 1
#endif

/*! \brief whether serialize using little endian */
#define NNDEPLOY_IO_NO_ENDIAN_SWAP (NNDEPLOY_LITTLE_ENDIAN == NNDEPLOY_IO_USE_LITTLE_ENDIAN)

namespace nndeploy {
namespace base {

/*!
 * \brief A generic inplace byte swapping function.
 * \param data The data pointer.
 * \param elem_bytes The number of bytes of the data elements
 * \param num_elems Number of elements in the data.
 * \note Always try pass in constant elem_bytes to enable
 *       compiler optimization
 */
inline void byteSwap(void* data, size_t elem_bytes, size_t num_elems) {
  for (size_t i = 0; i < num_elems; ++i) {
    uint8_t* bptr = reinterpret_cast<uint8_t*>(data) + elem_bytes * i;
    for (size_t j = 0; j < elem_bytes / 2; ++j) {
      uint8_t v = bptr[elem_bytes - 1 - j];
      bptr[elem_bytes - 1 - j] = bptr[j];
      bptr[j] = v;
    }
  }
}

}  // namespace base
}  // namespace nndeploy

#endif  // _NNDEPLOY_BASE_ENDIAN_H_

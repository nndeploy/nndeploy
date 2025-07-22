#if defined(__ANDROID__)
#include <sys/mman.h>
#define posix_madvise(addr, len, advice) madvise(addr, len, advice)
#endif
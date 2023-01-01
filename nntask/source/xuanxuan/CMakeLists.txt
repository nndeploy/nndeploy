# set
set(lib_target nntask_xuanxuan)
set(exe_target nndemo_xuanxuan)
set(lib_src)
set(exe_src)
set(obj)
set(lib)
set(exe)
set(dep)
set(link)

# include
include_directories(ROOT_PATH)

# lib
file(GLOB lib_src "*.cc" "*.h")
add_library(${lib_target} ${lib_src})

# exe
# if (NNKIT_ENABLE_NNDEMO)
#   file(GLOB exe_src "*.cpp")
#   add_executable(${exe_target} ${exe_src})
# endif()

# depend

# link

# install

# pack

# unset
unset(target)
unset(incs)
unset(srcs)
unset(deps)
unset(links)


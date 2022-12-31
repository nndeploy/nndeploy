# set
set(lib_target nndeploy)
set(lib_src)
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

# depend

# link

# install

# pack

# unset
unset(lib_target)
unset(lib_src)
unset(obj)
unset(lib)
unset(dep)
unset(link)


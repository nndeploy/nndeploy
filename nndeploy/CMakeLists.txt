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
add_subdirectory(source/base)
set(obj ${obj} "$<obj:base>")

add_subdirectory(source/device)
set(obj ${obj} "$<obj:device>")

add_subdirectory(source/inference)
set(obj ${obj} "$<obj:inference>")

add_subdirectory(source/audio)
set(obj ${obj} "$<obj:audio>")

add_subdirectory(source/vision)
set(obj ${obj} "$<obj:vision>")

add_subdirectory(source/engine)
set(obj ${obj} "$<obj:engine>")

add_library(${lib_target} ${obj})

# path


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


set_project("nndeploy", "nndeploy是一款模型端到端部署框架。以多端推理以及基于有向无环图模型部署为基础，致力为用户提供跨平台、简单易用、高性能的模型部署体验。")
set_version("2.0.0+0", {build = "%Y-%m-%d %H:%M"})
set_license("Apache-2.0")
add_rules("mode.debug", "mode.release")

add_repositories("local-repo xmake/repo")
includes("xmake/*.lua")

add_requires("tokenizer-cpp")  --需手动安装rust环境

if is_plat("linux") then 
    add_syslinks("pthread")
elseif is_plat("windows") then 
    add_cxflags("/IGNORE:4286 /wd4273 /wd4819")
    add_ldflags("/force:unresolved")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        add_defines("ENABLE_NNDEPLOY_BUILDING_DLL")
    end
end

if is_mode("debug") then 
    add_defines("NNDEPLOY_DEBUG")
end

if has_config("ENABLE_NNDEPLOY_SYMBOL_HIDE") then 
    set_symbols("hidden")
end

if has_config("ENABLE_NNDEPLOY_COVERAGE") then 
    add_cxxflags("clang::-fprofile-instr-generate -fcoverage-mapping")
    add_cxxflags("gxx::-coverage -fprofile-arcs -ftest-coverage")
    add_ldflags("gxx::-coverage -lgcov")
end

if has_config("ENABLE_NNDEPLOY_CXX11_ABI") then
    set_languages("c++11")
end

if has_config("ENABLE_NNDEPLOY_CXX14_ABI") then
    set_languages("c++14")
end

if has_config("ENABLE_NNDEPLOY_CXX17_ABI") then
    set_languages("c++17")
end

if has_config("ENABLE_NNDEPLOY_CXX20_ABI") then
    set_languages("c++20")
end

if has_config("ENABLE_NNDEPLOY_OPENMP") then
    add_defines("ENABLE_NNDEPLOY_OPENMP")
    add_requires("openmp")
end

if has_config("ENABLE_NNDEPLOY_ADDRESS_SANTIZER") then 
    set_policy("build.sanitizer.address", true)
end 

if has_config("ENABLE_NNDEPLOY_DOCS") then
    --TODO
end

if has_config("ENABLE_NNDEPLOY_TIME_PROFILER") then
    add_defines("ENABLE_NNDEPLOY_TIME_PROFILER")
end

if has_config("ENABLE_NNDEPLOY_OPENCV") then
    add_defines("ENABLE_NNDEPLOY_OPENCV")
    add_requires("opencv")
end

if has_config("ENABLE_NNDEPLOY_DEVICE_X86") then 
    add_defines("ENABLE_NNDEPLOY_DEVICE_X86")
end

if has_config("ENABLE_NNDEPLOY_DEVICE_ARM") then 
    add_defines("ENABLE_NNDEPLOY_DEVICE_ARM")
end

if has_config("ENABLE_NNDEPLOY_DEVICE_CUDA") then 
    add_requires("cuda")
end

if has_config("ENABLE_NNDEPLOY_DEVICE_METAL") or has_config("ENABLE_NNDEPLOY_DEVICE_APPLE_NPU") then 
    add_cflags("-fobjc-arc -Wno-shorten-64-to-32")
    add_cxxflags("-fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
    add_mxflags("x objective-c++ -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
end

if has_config("ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME") then 
    add_requires("onnxruntime 1.15.1")
end

set_installdir("dist")
target("nndeploy")
    set_kind("phony")
    add_rules("summary")
    add_deps("nndeploy_framework")


includes("@builtin/xpack")
xpack("nndeploy")
    add_targets("nndeploy_framework")
    set_basename("nndeploy")
    set_formats("zip")
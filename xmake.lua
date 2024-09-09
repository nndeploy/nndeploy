set_project("nndeploy", "nndeploy是一款模型端到端部署框架。以多端推理以及基于有向无环图模型部署为基础，致力为用户提供跨平台、简单易用、高性能的模型部署体验。")
set_version("2.0.0+0", {build = "%Y-%m-%d %H:%M"})
set_license("Apache-2.0")
add_rules("mode.debug", "mode.release")

includes("xmake/options.lua")
includes("xmake/summary.lua")
includes("xmake/plugin.lua")


set_languages("cxx11")

if is_plat("linux") then 
    add_syslinks("pthread")
elseif is_plat("windows") then 
    add_cxxflags("-Werror=return-type")
end

if has_config("ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME") then
    add_requires("onnxruntime 1.15.1")
end

if has_config("ENABLE_NNDEPLOY_OPENCV") then 
    add_requires("opencv")
    add_defines("ENABLE_NNDEPLOY_OPENCV")
end


target("nndeploy_framework")
    add_rules("summary")
    add_includedirs("framework/include")
    add_deps("nndeploy_plugin_basic")

    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end

    if has_config("ENABLE_NNDEPLOY_SYMBOL_HIDE") then 
        set_symbols("hidden")
    end

    if has_config("ENABLE_NNDEPLOY_COVERAGE") then 
        add_cxxflags("clang::-fprofile-instr-generate -fcoverage-mapping")
        add_cxxflags("gxx::-coverage -fprofile-arcs -ftest-coverage")
        add_ldflags("gxx::-coverage -lgcov")
    end

    if has_config("ENABLE_NNDEPLOY_OPENMP") then 
        add_defines("ENABLE_NNDEPLOY_OPENMP")
        add_requires("openmp")
    end

    if has_config("ENABLE_NNDEPLOY_ADDRESS_SANTIZER") then 
        set_policy("build.sanitizer.address", true)
    end 

    if has_config("ENABLE_NNDEPLOY_DOCS") then 
        --TODO: add docs
    end

    if has_config("ENABLE_NNDEPLOY_TIME_PROFILER") then 
        add_defines("ENABLE_NNDEPLOY_TIME_PROFILER")
    end

    if has_config("ENABLE_NNDEPLOY_OPENCV") then 
        add_packages("opencv",{public = true})
    end

    if has_config("ENABLE_NNDEPLOY_BASE") then 
        add_files("framework/source/nndeploy/base/*.cc")
        add_headerfiles("framework/include/(nndeploy/base/*.h)")
        add_headerfiles("framework/include/(nndeploy/base/*.hpp)")
    end

    if has_config("ENABLE_NNDEPLOY_THREAD_POOL") then 
        add_files("framework/source/nndeploy/thread_pool/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/thread_pool/*.h)")
    end

    if has_config("ENABLE_NNDEPLOY_CRYPTION") then 
        add_files("framework/source/nndeploy/cryption/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/cryption/*.h)")
    end

    if has_config("ENABLE_NNDEPLOY_DEVICE") then 
        add_files("framework/source/nndeploy/device/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/device/*.h)")

        if has_config("ENABLE_NNDEPLOY_DEVICE_CPU") then 
            add_files("framework/source/nndeploy/device/cpu/*.cc")
            -- add_headerfiles("framework/include/(nndeploy/device/cpu/*.h)")
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_ARM") then 
            add_defines("ENABLE_NNDEPLOY_DEVICE_ARM")
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_X86") then 
            add_defines("ENABLE_NNDEPLOY_DEVICE_X86")
        end

        -- 目前包仓库中，并未实现自动下载cuda包，需手动安装
        if has_config("ENABLE_NNDEPLOY_DEVICE_CUDA") then 
            add_requires("cuda")
            add_packages("cuda")
            add_defines("ENABLE_NNDEPLOY_DEVICE_CUDA")
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_CUDNN") then 
            -- add_requires("cudnn")  TODO
            -- add_packages("cudnn")  TODO
            add_defines("ENABLE_NNDEPLOY_DEVICE_CUDNN")
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_OPENCL") then 
            -- TODO
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_OPENGL") then 
            -- TODO
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_METAL") or has_config("ENABLE_NNDEPLOY_DEVICE_APPLE_NPU") then 
            add_cflags("-fobjc-arc -Wno-shorten-64-to-32")
            add_cxxflags("-fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
            add_mxflags("x objective-c++ -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_HVX") then 
            -- TODO
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_MTK_VPU") then 
            -- TODO
        end

        if has_config("ENABLE_NNDEPLOY_DEVICE_ASCEND_CL") then 
            add_files("framework/source/nndeploy/device/ascend_cl/*.cc")
            -- add_headerfiles("framework/include/(nndeploy/device/ascend_cl/*.h)")
        end
    end

    
    if has_config("ENABLE_NNDEPLOY_OP") then 
        add_files("framework/source/nndeploy/op/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/op/*.h)")

        local device_types = {
            cpu = "ENABLE_NNDEPLOY_DEVICE_CPU",
            x86 = "ENABLE_NNDEPLOY_DEVICE_X86",
            arm = "ENABLE_NNDEPLOY_DEVICE_ARM",
            cuda = "ENABLE_NNDEPLOY_DEVICE_CUDA",
            ascend_cl = "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL"
        }

        for device, config in pairs(device_types) do
            if has_config(config) then
                add_files("framework/source/nndeploy/op/" .. device .. "/*.cc")
                -- add_headerfiles("framework/include/(nndeploy/op/" .. device .. "/*.h)")
            end
        end
    end

    if has_config("ENABLE_NNDEPLOY_INTERPRET") then 
        -- TODO: CMAKE
        -- if (ENABLE_NNDEPLOY_INTERPRET)
        --     set(ENABLE_NNDEPLOY_PROTOBUF ON)
        --     set(ENABLE_NNDEPLOY_ONNX ON)
        -- endif()
    end

    if has_config("ENABLE_NNDEPLOY_NET") then 
        add_files("framework/source/nndeploy/net/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/net/*.h)")

        local device_types = {
            cpu = "ENABLE_NNDEPLOY_DEVICE_CPU",
            x86 = "ENABLE_NNDEPLOY_DEVICE_X86",
            arm = "ENABLE_NNDEPLOY_DEVICE_ARM",
            cuda = "ENABLE_NNDEPLOY_DEVICE_CUDA",
            ascend_cl = "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL"
        }

        for device, config in pairs(device_types) do
            if has_config(config) then
                add_files("framework/source/nndeploy/net/" .. device .. "/*.cc")
                -- add_headerfiles("framework/include/(nndeploy/net/" .. device .. "/*.h)")
            end
        end
    end

    if has_config("ENABLE_NNDEPLOY_INFERENCE") then 
        add_files("framework/source/nndeploy/inference/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/inference/*.h)")

        if has_config("ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME") then
            add_packages("onnxruntime")
            add_files("framework/source/nndeploy/inference/onnxruntime/*.cc")
            -- add_headerfiles("framework/include/nndeploy/inference/onnxruntime/*.h")
        end

        local inference_types = {
            tensorrt = "ENABLE_NNDEPLOY_INFERENCE_TENSORRT",
            tnn = "ENABLE_NNDEPLOY_INFERENCE_TNN",
            mnn = "ENABLE_NNDEPLOY_INFERENCE_MNN",
            openvino = "ENABLE_NNDEPLOY_INFERENCE_OPENVINO",
            coreml = "ENABLE_NNDEPLOY_INFERENCE_COREML",
            onnxruntime = "ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME",
            tflite = "ENABLE_NNDEPLOY_INFERENCE_TFLITE",
            ncnn = "ENABLE_NNDEPLOY_INFERENCE_NCNN",
            paddlelite = "ENABLE_NNDEPLOY_INFERENCE_PADDLELITE",
            rknn = "ENABLE_NNDEPLOY_INFERENCE_RKNN",
            ascend_cl = "ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL",
            snpe = "ENABLE_NNDEPLOY_INFERENCE_SNPE"
        }

        
        for inference, config in pairs(inference_types) do
            if has_config(config) then
                add_files("framework/source/nndeploy/inference/" .. inference .. "/*.cc")
                -- add_headerfiles("framework/include/nndeploy/inference/" .. inference .. "/*.h")
                if inference == "coreml" then
                    add_files("framework/source/nndeploy/inference/coreml/*.mm")  -- coreml 额外包含 .mm 文件
                end
                if inference == "onnxruntime" then
                    add_packages("onnxruntime")
                end
            end
        end
    end

    if has_config("ENABLE_NNDEPLOY_DAG") then 
        add_files("framework/source/nndeploy/dag/*.cc")
        -- add_headerfiles("framework/include/(nndeploy/dag/*.h)")
        -- add_headerfiles("framework/include/(nndeploy/dag/*.hpp)")
    end

    add_installfiles("framework/include/(nndeploy/**.h)", "framework/include/(nndeploy/**.hpp)",{prefixdir = "include"})

set_installdir("dist")

includes("@builtin/xpack")
xpack("nndeploy")
    add_targets("nndeploy_framework")
    set_basename("nndeploy")
    set_formats("zip")
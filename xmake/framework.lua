function wrap_add_files(module_name, tables)
    for name, config in pairs(tables) do
        if has_config(config) then
            local prefix = "$(projectdir)/framework/source"
            add_files(prefix .. "/nndeploy/" .. module_name .. "/" .. name .."/*.cc")
            add_headerfiles(prefix .. "/(nndeploy/" .. module_name .. "/" .. name .."/*.h)")
            add_headerfiles(prefix .. "/(nndeploy/" .. module_name .. "/" .. name .."/*.hpp)")
        end
    end
end


target("nndeploy_framework")
    add_includedirs("$(projectdir)/framework/include")

    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then
        set_kind("shared")
    else
        set_kind("static")
    end

    if has_config("ENABLE_NNDEPLOY_OPENMP") then
        add_packages("openmp")  --此处包描述内 已经完成了/openmp等flag的设置
    end

    if has_config("ENABLE_NNDEPLOY_OPENCV") then
        add_packages("opencv")
    end

    if has_config("ENABLE_NNDEPLOY_BASE") then 
        add_files("$(projectdir)/framework/source/nndeploy/base/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/base/*.h)")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/base/*.hpp)")
    end

    if has_config("ENABLE_NNDEPLOY_THREAD_POOL") then 
        add_files("$(projectdir)/framework/source/nndeploy/thread_pool/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/thread_pool/*.h)")
    end

    if has_config("ENABLE_NNDEPLOY_CRYPTION") then 
        add_files("$(projectdir)/framework/source/nndeploy/cryption/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/cryption/*.h)")
    end

    if has_config("ENABLE_NNDEPLOY_DEVICE") then 
        add_files("$(projectdir)/framework/source/nndeploy/device/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/device/*.h)")

        local device_tabels = {
            cpu = "ENABLE_NNDEPLOY_DEVICE_CPU",
            x86 = "ENABLE_NNDEPLOY_DEVICE_X86",
            arm = "ENABLE_NNDEPLOY_DEVICE_ARM",
            cuda = "ENABLE_NNDEPLOY_DEVICE_CUDA",
            -- cudnn = "ENABLE_NNDEPLOY_DEVICE_CUDNN",
            -- opencl = "ENABLE_NNDEPLOY_DEVICE_OPENCL",
            -- opengl = "ENABLE_NNDEPLOY_DEVICE_OPENGL",
            -- metal = "ENABLE_NNDEPLOY_DEVICE_METAL",
            -- apple_npu = "ENABLE_NNDEPLOY_DEVICE_APPLE_NPU",
            -- hvx = "ENABLE_NNDEPLOY_DEVICE_HVX",
            -- mtk_vpu = "ENABLE_NNDEPLOY_DEVICE_MTK_VPU",
            ascend_cl = "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL"
        }

        wrap_add_files("device", device_tabels)    
    end

    if has_config("ENABLE_NNDEPLOY_OP") then 
        add_files("$(projectdir)/framework/source/nndeploy/op/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/op/*.h)")

        local op_device_tabels = {
            cpu = "ENABLE_NNDEPLOY_DEVICE_CPU",
            x86 = "ENABLE_NNDEPLOY_DEVICE_X86",
            arm = "ENABLE_NNDEPLOY_DEVICE_ARM",
            cuda = "ENABLE_NNDEPLOY_DEVICE_CUDA",
            -- cudnn = "ENABLE_NNDEPLOY_DEVICE_CUDNN",
            -- opencl = "ENABLE_NNDEPLOY_DEVICE_OPENCL",
            -- opengl = "ENABLE_NNDEPLOY_DEVICE_OPENGL",
            -- metal = "ENABLE_NNDEPLOY_DEVICE_METAL",
            -- apple_npu = "ENABLE_NNDEPLOY_DEVICE_APPLE_NPU",
            -- hvx = "ENABLE_NNDEPLOY_DEVICE_HVX",
            -- mtk_vpu = "ENABLE_NNDEPLOY_DEVICE_MTK_VPU",
            ascend_cl = "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL"
        }

        wrap_add_files("op", op_device_tabels)  
    end

    if has_config("ENABLE_NNDEPLOY_INTERPRET") then
        --TODO:
    end

    if has_config("ENABLE_NNDEPLOY_NET") then
        add_files("$(projectdir)/framework/source/nndeploy/net/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/net/*.h)")

        local net_device_tabels = {
            cpu = "ENABLE_NNDEPLOY_DEVICE_CPU",
            x86 = "ENABLE_NNDEPLOY_DEVICE_X86",
            arm = "ENABLE_NNDEPLOY_DEVICE_ARM",
            cuda = "ENABLE_NNDEPLOY_DEVICE_CUDA",
            -- cudnn = "ENABLE_NNDEPLOY_DEVICE_CUDNN",
            -- opencl = "ENABLE_NNDEPLOY_DEVICE_OPENCL",
            -- opengl = "ENABLE_NNDEPLOY_DEVICE_OPENGL",
            -- metal = "ENABLE_NNDEPLOY_DEVICE_METAL",
            -- apple_npu = "ENABLE_NNDEPLOY_DEVICE_APPLE_NPU",
            -- hvx = "ENABLE_NNDEPLOY_DEVICE_HVX",
            -- mtk_vpu = "ENABLE_NNDEPLOY_DEVICE_MTK_VPU",
            ascend_cl = "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL"
        }

        wrap_add_files("net", net_device_tabels)
    end

    if has_config("ENABLE_NNDEPLOY_INFERENCE") then
        add_files("$(projectdir)/framework/source/nndeploy/inference/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/inference/*.h)")

        local inference_tabels = {
            tensorrt = "ENABLE_NNDEPLOY_INFERENCE_TENSORRT",
            openvino = "ENABLE_NNDEPLOY_INFERENCE_OPENVINO",
            coreml = "ENABLE_NNDEPLOY_INFERENCE_COREML",
            tflite = "ENABLE_NNDEPLOY_INFERENCE_TFLITE",
            onnxruntime = "ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME",
            ncnn = "ENABLE_NNDEPLOY_INFERENCE_NCNN",
            tnn = "ENABLE_NNDEPLOY_INFERENCE_TNN",
            mnn = "ENABLE_NNDEPLOY_INFERENCE_MNN",
            -- tvm = "ENABLE_NNDEPLOY_INFERENCE_TVM",
            paddlelite = "ENABLE_NNDEPLOY_INFERENCE_PADDLELITE",
            rknn = "ENABLE_NNDEPLOY_INFERENCE_RKNN",
            -- rknn_toolkit_1 = "ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1",
            -- rknn_toolkit_2 = "ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2",
            ascend_cl = "ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL",
            snpe = "ENABLE_NNDEPLOY_INFERENCE_SNPE",
        }
        wrap_add_files("inference", inference_tabels)

        if has_config("ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME") then
            add_packages("onnxruntime")
        end
    end

    if has_config("ENABLE_NNDEPLOY_DAG") then
        add_files("$(projectdir)/framework/source/nndeploy/dag/*.cc")
        add_headerfiles("$(projectdir)/framework/include/(nndeploy/dag/*.h)")
    end
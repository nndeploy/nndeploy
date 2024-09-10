target("nndeploy_plugin_basic")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/basic/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/basic/*.h")


target("nndeploy_plugin_infer")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/infer/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/infer/*.h")


target("nndeploy_plugin_codec")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/codec/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/codec/*.h")


target("nndeploy_plugin_detect")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/detect/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/detect/*.h")
    if has_config("ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR") then
        add_files("$(projectdir)/plugin/source/nndeploy/detect/detr/*.cc")
        add_headerfiles("$(projectdir)/plugin/include/nndeploy/detect/detr/*.h")
    end

    if has_config("ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO") then
        add_files("$(projectdir)/plugin/source/nndeploy/detect/yolo/*.cc")
        add_headerfiles("$(projectdir)/plugin/include/nndeploy/detect/yolo/*.h")
    end

target("nndeploy_plugin_segment")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/segment/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/segment/*.h")

    if has_config("ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING") then
        add_files("$(projectdir)/plugin/source/nndeploy/segment/segment_anything/*.cc")
        add_headerfiles("$(projectdir)/plugin/include/nndeploy/segment/segment_anything/*.h")
    end

target("nndeploy_plugin_tokenizer")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/tokenizer/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/tokenizer/*.h")

target("nndeploy_plugin_stable_diffusion")
    set_kind("$(kind)")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

    add_files("$(projectdir)/plugin/source/nndeploy/stable_diffusion/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/stable_diffusion/*.h")
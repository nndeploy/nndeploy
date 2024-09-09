target("nndeploy_plugin_basic")
    if has_config("ENABLE_NNDEPLOY_BUILD_SHARED") then 
        set_kind("shared")
    else
        set_kind("static")
    end

    add_files("$(projectdir)/plugin/source/nndeploy/basic/*.cc")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/framework/include")

target("nndeploy_plugin_torchscript")
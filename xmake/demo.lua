if has_config("ENABLE_NNDEPLOY_TEST") then
    add_requires("gflags",{system=false})
end

target("demo_dag")
    if not has_config("ENABLE_NNDEPLOY_DEMO") then
        set_default(false)
    end

    set_kind("binary")
    add_deps("nndeploy_framework")
    
    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/dag/**.cc")
    add_headerfiles("$(projectdir)/demo/dag/**.h")

    add_packages("opencv", "gflags", "onnxruntime")
    add_links("nndeploy_framework")
    add_ldflags("-Wl,--no-as-needed")


target("demo_detect")
    if not has_config("ENABLE_NNDEPLOY_DEMO") then
        set_default(false)
    end

    set_kind("binary")
    add_deps("nndeploy_framework")

    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/detect/**.cc")
    add_headerfiles("$(projectdir)/demo/detect/**.h")

    add_packages("opencv", "gflags", "onnxruntime")
    add_links("nndeploy_framework","nndeploy_plugin_codec")
    add_ldflags("-Wl,--no-as-needed")

target("demo_segment")
    if not has_config("ENABLE_NNDEPLOY_DEMO") then
        set_default(false)
    end

    set_kind("binary")
    add_deps("nndeploy_framework")

    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/segment/**.cc")
    add_headerfiles("$(projectdir)/demo/segment/**.h")

    add_packages("opencv", "gflags", "onnxruntime")
    add_links("nndeploy_framework", "nndeploy_plugin_codec")
    add_ldflags("-Wl,--no-as-needed")

target("demo_test_net")
    if not has_config("ENABLE_NNDEPLOY_DEMO") then
        set_default(false)
    end

    set_kind("binary")
    add_deps("nndeploy_framework")

    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/test_net/**.cc")
    add_headerfiles("$(projectdir)/demo/test_net/**.h")

    add_packages("opencv", "gflags", "onnxruntime")
    add_links("nndeploy_framework" )
    add_ldflags("-Wl,--no-as-needed")

target("demo_test_op")
    if not has_config("ENABLE_NNDEPLOY_DEMO") then
        set_default(false)
    end

    set_kind("binary")
    add_deps("nndeploy_framework")

    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/test_op/**.cc")
    add_headerfiles("$(projectdir)/demo/test_op/**.h")

    add_packages("opencv", "gflags")
    add_links("nndeploy_framework" )
    add_ldflags("-Wl,--no-as-needed")

-- target("demo_tokenizer_cpp")
--     if not has_config("ENABLE_NNDEPLOY_DEMO") then
--         set_default(false)
--     end

--     set_kind("binary")
--     add_deps("nndeploy_framework")

--     add_includedirs("$(projectdir)/framework/include")
--     add_includedirs("$(projectdir)/plugin/include")
--     add_includedirs("$(projectdir)/demo")

--     add_files("$(projectdir)/demo/*.cc")
--     add_headerfiles("$(projectdir)/demo/*.h")

--     add_files("$(projectdir)/demo/tokenizer_cpp/**.cc")
--     add_headerfiles("$(projectdir)/demo/tokenizer_cpp/**.h")

--     add_packages("opencv", "gflags","tokenizer-cpp")
--     add_links("nndeploy_framework" )
--     add_ldflags("-Wl,--no-as-needed")
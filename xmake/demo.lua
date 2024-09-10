
add_requires("gflags")
target("demo_dag")
    set_kind("binary")
    add_deps("nndeploy_framework")
    
    add_includedirs("$(projectdir)/framework/include")
    add_includedirs("$(projectdir)/plugin/include")
    add_includedirs("$(projectdir)/demo")

    add_files("$(projectdir)/demo/*.cc")
    add_headerfiles("$(projectdir)/demo/*.h")

    add_files("$(projectdir)/demo/dag/*.cc")

    add_files("$(projectdir)/plugin/source/nndeploy/detect/*.cc")
    add_headerfiles("$(projectdir)/plugin/include/nndeploy/detect/*.h")

    add_packages("opencv")
    add_packages("gflags")
    add_links("nndeploy_framework")
    add_packages("onnxruntime")
    add_ldflags("-Wl,--no-as-needed")
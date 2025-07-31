
function(smart_force_load_libraries target_name)
  if (APPLE)
    target_link_options(${target_name} PRIVATE 
      "LINKER:-undefined,dynamic_lookup"
    )
  elseif (UNIX AND NOT WIN32)
    target_link_options(${target_name} PRIVATE 
      "LINKER:--no-as-needed" 
    )
  elseif(WIN32 AND MSVC)
    # Windows下只对静态库使用/WHOLEARCHIVE，动态库正常链接
    # foreach(lib ${NNDEPLOY_PLUGIN_LIST})
    #   if(TARGET ${lib})
    #     get_target_property(lib_type ${lib} TYPE)
    #     if(lib_type STREQUAL "STATIC_LIBRARY")
    #       target_link_libraries(${target_name} PRIVATE 
    #         /WHOLEARCHIVE:${lib}
    #       )
    #     else()
    #       target_link_libraries(${target_name} PRIVATE ${lib})
    #     endif()
    #   else()
    #     # 对于非目标库文件，检查扩展名
    #     get_filename_component(lib_ext ${lib} EXT)
    #     if(lib_ext STREQUAL ".lib")
    #       # 检查是否是DLL的导入库（通常导入库较小）
    #       # 这里简化处理，直接正常链接
    #       target_link_libraries(${target_name} PRIVATE ${lib})
    #     else()
    #       target_link_libraries(${target_name} PRIVATE ${lib})
    #     endif()
    #   endif()
    # endforeach()
  elseif(WIN32 AND MINGW)
    target_link_options(${target_name} PRIVATE 
      "LINKER:--no-as-needed"
    )
  endif()
endfunction()
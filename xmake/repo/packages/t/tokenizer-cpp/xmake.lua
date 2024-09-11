package("tokenizer-cpp")
    set_homepage("https://github.com/mlc-ai/tokenizers-cpp")
    set_description("Universal cross-platform tokenizers binding to HF and sentencepiece")
    set_license("Apache-2.0")

    --add_urls("https://github.com/mlc-ai/tokenizers-cpp/archive/refs/tags/v$(version).tar.gz",
    --"https://github.com/mlc-ai/tokenizers-cpp.git")
    --add_versions("0.1.0", "36e1f3783e35997fdc911d45d6aab17c485ddb03df83f12ae5db31a493a5a2ab")
    add_urls("https://github.com/mlc-ai/tokenizers-cpp.git")
    add_versions("2024.09.11", "5de6f656c06da557d4f0fb1ca611b16d6e9ff11d")

    add_configs("msgpack_use_boost", {description = "Use Boost library.", default = false, type = "boolean"})
    add_configs("mlc_enable_sentencepiece_tokenizer", {description = "Enable SentencePiece tokenizer", default = false, type = "boolean"})
    add_configs("spm_enable_shared", {description = "override sentence piece config", default = false, type = "boolean"})
    add_configs("spm_enable_tcmalloc", {description = "override sentence piece config", default = false, type = "boolean"})

    add_deps("cmake")
    --add_deps("msgpack-c", "sentencepiece")

    if is_plat("linux") then
        add_syslinks("pthread", "dl")
    end

    on_install(function (package)
        --io.replace("CMakeLists.txt", "add_subdirectory(msgpack)", "", {plain = true})
        --io.replace("CMakeLists.txt", "add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)", "", {plain = true})

        local configs = {"-DCMAKE_CXX_STANDARD=17"}
        table.insert(configs, "-DMSGPACK_USE_BOOST=" .. (package:config("msgpack_use_boost") and "ON" or "OFF"))
        table.insert(configs, "-DMLC_ENABLE_SENTENCEPIECE_TOKENIZER=" .. (package:config("mlc_enable_sentencepiece_tokenizer") and "ON" or "OFF"))
        table.insert(configs, "-DSPM_ENABLE_SHARED=" .. (package:config("spm_enable_shared") and "ON" or "OFF"))
        table.insert(configs, "-DSPM_ENABLE_TCMALLOC=" .. (package:config("spm_enable_tcmalloc") and "ON" or "OFF"))
        -- import("package.tools.cmake").install(package, configs,{packagedeps = {"sentencepiece","msgpack-c"}})
        import("package.tools.cmake").install(package, configs)

        os.cp("include/*.h", package:installdir("include"))
        os.cp(path.join(package:buildir(), "*.a"), package:installdir("lib"))
    end)
    
    add_linkgroups("tokenizers_cpp", "tokenizers_c", {group = true})

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <iostream>
            
            static void test() {
                std::unique_ptr<tokenizers::Tokenizer> tokenizer = tokenizers::Tokenizer::FromBlobJSON("{}");
                std::cout << "Tokenizer successfully created!" << std::endl;
                std::vector<int32_t> encoded = tokenizer->Encode("Hello, world!");
                std::cout << "Encoded size: " << encoded.size() << std::endl;
            }
        ]]}, {configs = {languages = "c++17"}, includes = "tokenizers_cpp.h"}))
    end)

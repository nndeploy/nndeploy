rule("summary")
    before_build(function (target)
        import("core.project.project")
        import("core.project.config")
        print("-------------------------- Summary --------------------------")
        print("  Version : ", project.version())
        print("  System : ", os.host())
        print("  Architecture : ", os.arch())
        print("  Build type : ", config.mode())

        local options = project.options()
        local optionnames = {}
        for optionname, _ in pairs(options) do
            table.insert(optionnames, optionname)
        end

        table.sort(optionnames)

        local max_length = 0
        for _, optionname in ipairs(optionnames) do
            max_length = math.max(max_length, #optionname)
        end
        local padding_length = max_length + 3

        for _, optionname in ipairs(optionnames) do
            local option = options[optionname]
            print(string.format("  %-" .. padding_length .. "s: %s", optionname, option:enabled() and "ON" or "OFF"))
        end
        print("-------------------------------------------------------------")
    end)
rule_end()
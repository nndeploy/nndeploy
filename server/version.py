CONFIG = {
    "schema": 1,

    "default_provider": {
        "frontend":  {"owner": "nndeploy", "repo": "nndeploy_frontend"},
        "templates": {"owner": "nndeploy", "repo": "nndeploy-workflow"}
    },

    "versions": {
        "2.6.1": {
            "frontend":  {"tag": "v1.4.0", "asset": "dist.zip"},
            "templates": {"tag": "v1.0.0", "asset": "nndeploy-workflow.zip"}
        },
        "2.6.2": {
            "frontend":  {"tag": "v1.4.2", "asset": "dist.zip"},
            "templates": {"tag": "v1.0.1", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.0": {
            "frontend":  {"tag": "v1.5.0", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.1": {
            "frontend":  {"tag": "v1.5.0", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.2": {
            "frontend":  {"tag": "v1.5.1", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.3": {
            "frontend":  {"tag": "v1.5.1", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.4": {
            "frontend":  {"tag": "v1.5.1", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.5": {
            "frontend":  {"tag": "v1.5.2", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.6": {
            "frontend":  {"tag": "v1.5.3", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        },
        "3.0.7": {
            "frontend":  {"tag": "v1.5.4", "asset": "dist.zip"},
            "templates": {"tag": "v1.1.0", "asset": "nndeploy-workflow.zip"}
        }
    },

    "ranges": [],

    "fallback": {
        "frontend":  {"tag": "latest", "asset": "dist.zip"},
        "templates": {"tag": "latest", "asset": "nndeploy-workflow.zip"}
    }
}

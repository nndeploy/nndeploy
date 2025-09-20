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
        }
    },
    
    "versions": {
        "2.6.2": {
            "frontend":  {"tag": "v1.4.2", "asset": "dist.zip"},
            "templates": {"tag": "v1.0.1", "asset": "nndeploy-workflow.zip"}
        }
    },

    "ranges": [],

    "fallback": {
        "frontend":  {"tag": "latest", "asset": "dist.zip"},
        "templates": {"tag": "latest", "asset": "dist.zip"}
    }
}

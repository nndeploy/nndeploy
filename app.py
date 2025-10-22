# python app.py
# This is the entry point script for the nndeploy server
# 
# Features:
# This script starts a complete nndeploy server with the following features:
# - Start Web server (default listening on 0.0.0.0:8888)
# - Manage task queue and worker processes
# - Handle plugin loading and management
# - Provide logging and monitoring
# - Support frontend interface services
#
# Usage:
# python app.py                                    # Start server with default configuration
# python app.py --port 9000                        # Specify port
# python app.py --host 127.0.0.1 --port 9000       # Specify host and port
# python app.py --resources ./my_resources         # Specify resource directory
# python app.py --log ./my_logs/server.log         # Specify log file path
# python app.py --plugin plugin1.py plugin2.so     # Load specified plugins
# python app.py --front-end-version @latest        # Specify frontend version

import nndeploy.server.app as app

if __name__ == "__main__":
    app.main()

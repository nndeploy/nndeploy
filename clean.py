# python clean.py
# This is an entry point for the cleanup script that cleans nndeploy server directories
# 
# Features:
# This script can clean the following directories:
# - logs: logs directory
# - template: resource template directory (resources/template)
# - db: database directory (resources/db) 
# - frontend: frontend directory
# - plugin: plugin directory
#
# Usage:
# python clean.py                    # clean all directories
# python clean.py --logs             # only clean logs directory
# python clean.py --template         # only clean template directory
# python clean.py --db               # only clean database directory
# python clean.py --frontend         # only clean frontend directory
# python clean.py --plugin           # only clean plugin directory
# python clean.py --keep logs db     # keep logs and db directory, clean other directories

import nndeploy.server.clean as clean

if __name__ == "__main__":
    clean.main()

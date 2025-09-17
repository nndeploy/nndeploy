
# python run_json.py
# This is the entry point script for the nndeploy graph runner
# 
# Features:
# This script provides a complete graph runner with the following features:
# - Load and run computational graphs from JSON files
# - Support custom input and output path configuration
# - Support dynamic node parameter configuration
# - Support plugin loading and management
# - Support parallel execution type configuration
# - Provide debugging and dump functionality
#
# Usage:
# python run_json.py --json_file graph.json                           # Run graph with default configuration
# python run_json.py --json_file graph.json --name my_graph           # Specify graph name
# python run_json.py --json_file graph.json --task_id task001         # Specify task ID
# python run_json.py --json_file graph.json --node_param node_name1:key1:value1   # Set node parameters
# python run_json.py --json_file graph.json --parallel_type sequential # Set parallel type
# python run_json.py --json_file graph.json --dump                    # Enable dump mode
# python run_json.py --json_file graph.json --plugin plugin1.py       # Load specified plugin


import nndeploy.dag.run_json as run_json

if __name__ == "__main__":
    run_json.main()

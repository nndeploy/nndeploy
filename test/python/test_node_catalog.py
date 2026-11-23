"""Optional node dependencies must not break the editor's node catalog."""
import json
import unittest
from unittest.mock import patch

from nndeploy.dag import node


class NodeCatalogTests(unittest.TestCase):
    def test_missing_dependency_keeps_available_nodes(self):
        def describe(key):
            if key == "example.Optional":
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return json.dumps({"key_": key})

        with patch.object(node, "import_global_import_lib"), \
                patch.object(node, "get_node_keys", return_value=["example.Optional", "example.Available"]), \
                patch.object(node, "get_node_json", side_effect=describe), \
                self.assertLogs(node.__name__, level="WARNING") as logs:
            catalog = json.loads(node.get_all_node_json())
        leaves = [item["id"] for item in catalog["nodes"] if item["type"] == "leaf"]
        self.assertEqual(leaves, ["example.Available"])
        self.assertIn("torch", logs.output[0])

    def test_unexpected_errors_are_not_hidden(self):
        with patch.object(node, "import_global_import_lib"), \
                patch.object(node, "get_node_keys", return_value=["example.Broken"]), \
                patch.object(node, "get_node_json", side_effect=ValueError("invalid node")):
            with self.assertRaisesRegex(ValueError, "invalid node"):
                node.get_all_node_json()


if __name__ == "__main__":
    unittest.main()

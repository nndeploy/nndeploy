"""Ordered checkpoint renaming, shared by component and adapter loaders."""
import re

def map_weight_name(name, rules):
    for pattern, replacement in rules.items():
        if not isinstance(replacement, str):
            raise ValueError('This loader requires one-to-one weight mappings')
        name = re.sub(pattern, replacement, name)
    return name

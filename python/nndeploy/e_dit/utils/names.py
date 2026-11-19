"""Hierarchical checkpoint names."""
def add_prefix(name, prefix):
    return f'{prefix}.{name}' if prefix else name

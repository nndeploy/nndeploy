"""Component declarations and observable loading results."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    required: bool = True
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Component name must be nonempty')
        if self.name in self.dependencies:
            raise ValueError(f'{self.name}: component cannot depend on itself')

    def to_dict(self):
        return asdict(self)


def order_components(specs):
    """Stable dependency order; reject invalid plans before loading weights."""
    specs = tuple(specs)
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError('Duplicate component declarations')
    ordered, active, done = [], set(), set()
    def visit(name):
        if name not in by_name:
            raise ValueError(f'Undeclared component dependency: {name}')
        if name in active:
            raise ValueError(f'Cyclic component dependency: {name}')
        if name in done:
            return
        active.add(name)
        for dependency in by_name[name].dependencies:
            visit(dependency)
        active.remove(name)
        done.add(name)
        ordered.append(by_name[name])
    for spec in specs:
        visit(spec.name)
    return tuple(ordered)


@dataclass(frozen=True)
class ComponentLoadReport:
    name: str
    source: str | None
    status: str
    elapsed_s: float = 0.0
    parameter_count: int = 0
    dtypes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self):
        return asdict(self)

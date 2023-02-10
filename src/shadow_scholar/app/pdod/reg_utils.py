from typing import Callable, Dict, Generic, TypeVar, Tuple

T = TypeVar("T")


class CallableRegistry(Generic[T]):
    """Singleton registry for datasets."""

    callables: Dict[str, Callable[..., T]]

    def __init__(self):
        self.callables = {}

    def add(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(func: Callable[..., T]):
            if name in self.callables:
                raise KeyError(f"Dataset {name} already in the registry")
            self.callables[name] = func
            return func

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        return self.callables[name]

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.callables.keys())

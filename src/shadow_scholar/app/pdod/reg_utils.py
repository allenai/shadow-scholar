from functools import partial
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")
V = TypeVar("V")


class CallableRegistry(Generic[T]):
    """Singleton registry for datasets."""

    callables: Dict[str, Union[Callable[..., T], Type[T]]]

    def __init__(self):
        self.callables = {}

    @overload
    def _decorator(  # pyright: ignore
        self, func: Callable[..., T], func_name: Optional[str] = None
    ) -> Callable[..., T]:
        ...

    @overload
    def _decorator(
        self, func: Type[T], func_name: Optional[str] = None
    ) -> Type[T]:
        ...

    def _decorator(
        self,
        func: Union[Callable[..., T], Type[T]],
        func_name: Optional[str] = None,
    ) -> Union[Callable[..., T], Type[T]]:
        name = func_name or func.__name__
        if name in self.callables:
            raise KeyError(f"Dataset {name} already in the registry")
        self.callables[name] = func
        return func

    def add(self, name: str) -> Callable[[V], V]:
        return partial(self._decorator, func_name=name)  # type: ignore

    def get(self, name: str) -> Callable[..., T]:
        return self.callables[name]

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.callables.keys())

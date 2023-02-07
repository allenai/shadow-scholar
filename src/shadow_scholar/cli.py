from functools import partial
import inspect
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar
)

from necessary import necessary
from typing_extensions import ParamSpec

# parameter spec
PS = ParamSpec("PS")

# return type
RT = TypeVar("RT")

# generic type
T = TypeVar("T")

# sentinel value
MISSING = object()


class Argument:
    """Holds parameters for argparse.ArgumentParser.add_argument."""

    args: Tuple[str, ...]
    kwargs: Dict[str, Any]

    def __init__(
        self,
        *name_or_flags: str,
        action: Optional[str] = MISSING,  # type: ignore
        nargs: Optional[str] = MISSING,  # type: ignore
        const: Optional[str] = MISSING,  # type: ignore
        default: Optional[T] = MISSING,  # type: ignore
        type: Optional[Type[T]] = MISSING,  # type: ignore
        choices: Optional[List[str]] = MISSING,  # type: ignore
        required: Optional[bool] = MISSING,  # type: ignore
        help: Optional[str] = MISSING,  # type: ignore
        metavar: Optional[str] = MISSING,  # type: ignore
        dest: Optional[str] = MISSING,  # type: ignore
    ):
        self.args = name_or_flags
        self.kwargs = {
            **({"action": action} if action is not MISSING else {}),
            **({"nargs": nargs} if nargs is not MISSING else {}),
            **({"const": const} if const is not MISSING else {}),
            **({"default": default} if default is not MISSING else {}),
            **({"type": type} if type is not MISSING else {}),
            **({"choices": choices} if choices is not MISSING else {}),
            **({"required": required} if required is not MISSING else {}),
            **({"help": help} if help is not MISSING else {}),
            **({"metavar": metavar} if metavar is not MISSING else {}),
            **({"dest": dest} if dest is not MISSING else {}),
        }


class EntryPoint(Generic[PS, RT]):
    def __init__(
        self,
        name: str,
        func: Callable[PS, RT],
        args: List[Argument],
        reqs: List[str],
    ):
        self.name = name
        self.func = func
        self.args = args
        self.reqs = reqs

        # check that all requirements are met (error raised later if not)
        self.missing_reqs = [r for r in reqs if not necessary(r, soft=True)]

    def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> RT:
        """Run the function."""

        if self.missing_reqs:
            raise ModuleNotFoundError(
                f"Missing requirements: {', '.join(self.missing_reqs)}"
            )
        return self.func(*args, **kwargs)

    def cli(self) -> RT:
        """Run the function from the command line."""
        ap = ArgumentParser(f"shadow-scholar {self.name}")
        for arg in self.args:
            ap.add_argument(*arg.args, **arg.kwargs)
        opts, *_ = ap.parse_known_args()

        parsed_args = inspect.signature(self.func).bind(vars(opts)).arguments
        return self.func(**parsed_args)  # pyright: ignore

    @classmethod
    def decorate(
        cls,
        func: Callable[PS, RT],
        name: Optional[str] = None,
        arguments: Optional[List[Argument]] = None,
        requirements: Optional[List[str]] = None,
    ) -> 'EntryPoint[PS, RT]':
        """Decorator designed to add function to a registry alongside
        all its arguments and requirements.

        We add the requirement for future parsing, rather than
        building a ArgumentParser here, for two reasons:
        1. We want to be able for all decorated functions to be able to
            use overlapping arguments, and
        2. Creating a single parser here might lead to a lot of unintended
            side effects.

        Args:
            func (Callable): The function to be decorated.
            name (str, optional): Name to use for the function when it is
                called from the command line. If none is provided, the
                function name is used. Defaults to None.
            arguments (List[Argument], optional): A list of Argument objects
                to be passed to the ArgumentParser. The available options
                are the same as those for argparse.ArgumentParser.add_argument.
                Defaults to None.
            requirements (Optional[List[str]], optional): A list of required
                packages for the function to run. Defaults to None.
        """

        name = name or func.__name__
        arguments = arguments or []
        func_requirements = requirements or []

        # create the entry point by wrapping the function
        entry_point = cls(
            name=name,
            func=func,
            args=arguments,
            reqs=func_requirements,
        )

        # add the function to the registry
        Registry().add(entry_point)

        return entry_point


class Registry:
    """A registry to hold all the functions decorated with @cli."""

    def __new__(cls):
        """Singleton pattern for the registry."""
        if not hasattr(cls, "__instance__"):
            cls.__instance__ = super(Registry, cls).__new__(cls)
            cls.__instance__.__init__()
        return cls.__instance__

    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    def add(self, entry_point: EntryPoint) -> None:
        """Add an entry point to the registry."""
        if entry_point.name in self._registry:
            raise KeyError(f"Func {entry_point.name} already in the registry")

        self._registry[entry_point.name] = entry_point

    def cli(
        self,
        name: Optional[str] = None,
        arguments: Optional[List[Argument]] = None,
        requirements: Optional[List[str]] = None,
    ) -> Callable[[Callable[PS, RT]], 'EntryPoint[PS, RT]']:
        """A decorator to add a function to the registry.

        Args:
            name (str, optional): Name to use for the function
                when it is called from the command line. If none is provided,
                the function name is used. Defaults to None.
            arguments (List[Argument], optional): A list of Argument objects
                to be passed to the ArgumentParser. The available options
                are the same as those for argparse.ArgumentParser.add_argument.
                Defaults to None.
            requirements (Optional[List[str]], optional): A list of required
        """
        decorated = partial(
            EntryPoint.decorate,    # type: ignore
            name=name,
            arguments=arguments,
            requirements=requirements
        )
        return decorated   # type: ignore

    def run(self):
        """Creates a click command group for all registered functions."""
        parser = ArgumentParser("shadow-scholar")
        parser.add_argument(
            "entrypoint", choices=self._registry.keys()
        )
        opts, _ = parser.parse_known_args()

        if opts.entrypoint in self._registry:
            return self._registry[opts.entrypoint]()

        raise ValueError(f"No entrypoint found for {opts.entrypoint}")


@contextmanager
def safe_import():
    """Context manager to safely import a package.

    Args:
        package (str): Name of the package to import.
    """
    try:
        yield
    except (ModuleNotFoundError, ImportError):
        pass


run = Registry().run
cli = Registry().cli

__all__ = ["run", "cli", "Argument", "safe_import"]

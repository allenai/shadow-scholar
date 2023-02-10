import inspect
import json
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
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
M = object()


# sentinel value for missing arguments
class _M:
    ...  # noqa: E701


M = _M()  # noqa: E305


class Argument:
    """Holds parameters for argparse.ArgumentParser.add_argument."""

    args: Tuple[str, ...]
    kwargs: Dict[str, Any]

    def __init__(
        self,
        *name_or_flags: str,
        action: Optional[str] = M,  # type: ignore
        nargs: Optional[str] = M,  # type: ignore
        const: Optional[str] = M,  # type: ignore
        default: Optional[T] = M,  # type: ignore
        type: Union[Type[T], Callable[..., T], None] = _M,  # type: ignore
        choices: Optional[List[str]] = M,  # type: ignore
        required: Optional[bool] = M,  # type: ignore
        help: Optional[str] = M,  # type: ignore
        metavar: Optional[str] = M,  # type: ignore
        dest: Optional[str] = M,  # type: ignore
    ):
        self.args = name_or_flags
        self.kwargs = {
            **({"action": action} if action is not M else {}),
            **({"nargs": nargs} if nargs is not M else {}),
            **({"const": const} if const is not M else {}),
            **({"default": default} if default is not M else {}),
            **({"type": type} if type is not _M else {}),
            **({"choices": choices} if choices is not M else {}),
            **({"required": required} if required is not M else {}),
            **({"help": help} if help is not M else {}),
            **({"metavar": metavar} if metavar is not M else {}),
            **({"dest": dest} if dest is not M else {}),
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

    def cli(self, args: Optional[List[str]] = None) -> RT:
        """Run the function from the command line."""
        ap = ArgumentParser(f"shadow-scholar {self.name}")
        for arg in self.args:
            ap.add_argument(*arg.args, **arg.kwargs)

        opts, *_ = ap.parse_known_args(args)

        parsed_args = inspect.signature(self.func).bind(**vars(opts)).arguments
        return self.func(**parsed_args)  # pyright: ignore

    @classmethod
    def decorate(
        cls,
        func: Callable[PS, RT],
        name: Optional[str] = None,
        arguments: Optional[List[Argument]] = None,
        requirements: Optional[List[str]] = None,
    ) -> "EntryPoint[PS, RT]":
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

    __instance__: "Registry"
    _registry: Dict[str, "EntryPoint"]

    def __new__(cls):
        """Singleton pattern for the registry."""
        if not hasattr(cls, "__instance__"):
            cls.__instance__ = super(Registry, cls).__new__(cls)
        return cls.__instance__

    def __init__(self) -> None:
        if not hasattr(self, "_registry"):
            self._registry = {}

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
    ) -> Callable[[Callable[PS, RT]], "EntryPoint[PS, RT]"]:
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
            EntryPoint.decorate,  # type: ignore
            name=name,
            arguments=arguments,
            requirements=requirements,
        )
        return decorated  # type: ignore

    def run(self):
        """Creates a click command group for all registered functions."""
        parser = ArgumentParser("shadow-scholar")
        parser.add_argument("entrypoint", choices=self._registry.keys())

        # stop at the first argument that does not start with a dash
        # the +2 is needed because sys.argv[0] is the script name (that
        # accounts for the first +1) and slicing is exclusive (that accounts
        # for the second +1).
        try:
            i = [arg.startswith('-') for arg in sys.argv[1:]].index(False)
        except ValueError:
            i = len(sys.argv)
        args = sys.argv[1:i]

        opts, rest = parser.parse_known_args(args)
        rest += sys.argv[i:]

        if opts.entrypoint in self._registry:
            return self._registry[opts.entrypoint].cli(rest)

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


def load_kwargs(path_or_json: str) -> Dict[str, Any]:
    if Path(path_or_json).exists():
        with open(path_or_json) as f:
            path_or_json = f.read()
    return json.loads(path_or_json)


run = Registry().run
cli = Registry().cli

__all__ = ["run", "cli", "Argument", "safe_import", "load_kwargs"]

import inspect
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from necessary import necessary
from typing_extensions import Concatenate, ParamSpec

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


class Registry:
    """A registry to hold all the functions decorated with @cli."""

    def __new__(cls):
        """Singleton pattern for the registry."""
        if not hasattr(cls, "__instance__"):
            cls.__instance__ = super(Registry, cls).__new__(cls)
            cls.__instance__.__init__()
        return cls.__instance__

    def __init__(self) -> None:
        self._callable_registry: Dict[str, Callable] = {}
        self._requirements_registry: Dict[str, List[str]] = {}

    def cli(
        self,
        name: Optional[str] = None,
        arguments: Optional[List[Argument]] = None,
        requirements: Optional[List[str]] = None,
        entrypoint: bool = True,
    ):
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
            entrypoint (bool, optional): _description_. Defaults to True.
        """

        def decorator(
            func: Callable[Concatenate[PS], RT]
        ) -> Callable[Concatenate[PS], RT]:
            """Decorator designed to add function to a registry alongside
            all its arguments and requirements.

            We add the requirement for future parsing, rather than
            building a ArgumentParser here, for two reasons:
            1. We want to be able for all decorated functions to be able to
               use overlapping arguments, and
            2. Creating a single parser here might lead to a lot of unintended
               side effects.
            """

            func_arg_name = name or func.__name__
            func_arguments = arguments or []
            func_requirements = requirements or []
            is_entrypoint = entrypoint

            # check that all requirements are met
            missing_requirements = [
                req
                for req in func_requirements
                if not necessary(req, soft=True)
            ]

            def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> RT:
                ap = ArgumentParser(f"shadow-scholar {func_arg_name}")
                for arg in func_arguments:
                    ap.add_argument(*arg.args, **arg.kwargs)

                if missing_requirements:
                    print("The following requirements are missing:", end=" ")
                    print(" ".join(missing_requirements))
                    sys.exit(1)

                opts, *_ = ap.parse_known_args()

                parsed_args = (
                    inspect.signature(func)
                    .bind(*args, **{**vars(opts), **kwargs})
                    .arguments
                )

                return func(**parsed_args)  # pyright: ignore

            if is_entrypoint and func_arg_name in self._callable_registry:
                raise ValueError(f"Entry point {func_arg_name} already exists")
            elif is_entrypoint:
                self._callable_registry[func_arg_name] = wrapper
            return func

        return decorator

    def run(self):
        """Creates a click command group for all registered functions."""
        parser = ArgumentParser("shadow-scholar")
        parser.add_argument(
            "entrypoint", choices=self._callable_registry.keys()
        )
        opts, _ = parser.parse_known_args()

        if opts.entrypoint in self._callable_registry:
            return self._callable_registry[opts.entrypoint]()

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

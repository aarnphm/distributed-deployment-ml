import dataclasses as dc
import functools
import inspect
import threading
import typing as t

# https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md
_T = t.TypeVar("_T")

assert_msg: str = """\
{classname} does not support batch inference, meanwhile 'run_batch' is implemented.
"""


def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: t.Tuple[t.Union[type, t.Callable[..., t.Any]], ...] = (()),
) -> t.Callable[[_T], _T]:
    # If used within a stub file, the following implementation can be
    # replaced with "...".
    return lambda a: a


def _get_local_methods_name(cls: t.Any, exclude: t.Iterable[str] = ()) -> t.Tuple[str]:
    """Get method name of a class, excluding classmethod and staticmethod"""
    _methods = set()
    for m in cls.__dict__:
        if callable(cls.__dict__[m]) and not inspect.isclass(cls.__dict__[m]):
            mtype = type(cls.__dict__[m])
            if mtype != classmethod and mtype != staticmethod:
                _methods.add(m)
    return tuple(_methods.difference(set(exclude)))


@dc.dataclass
class _RunnableInternalState:
    is_initialized: bool = False
    in_setup: bool = False
    setup_called: bool = False

    def reset(self):
        """reset transient state. This should be called after every method implementation, so
        the only method-dependent attributes is reset"""
        self.in_setup = False

    def export(self):
        return _RunnableInternalState(
            in_setup=self.in_setup,
            setup_called=self.setup_called,
            is_initialized=self.is_initialized,
        )


_uninitialized_runnable_state = _RunnableInternalState()


def wrap_method_once(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """manage Runnable state for given method"""
    # we don't re-wrap methods that had the state management wrapper
    if hasattr(func, 'handler_wrapped'):
        return func

    @functools.wraps(func)
    def wrapped_runnable_method(*args, **kw):
        # check the first args, if it is self, otherwise call the wrapped function
        # we might wrapped a callable that is not a method
        if args and isinstance(args[0], Runnable):
            self, args = args[0], args[1:]
            return self._call_wrapped_method(func, *args, **kw)
        else:
            return func(*args, **kw)

    wrapped_runnable_method.handler_wrapped = True
    return wrapped_runnable_method


if t.TYPE_CHECKING:

    @__dataclass_transform__()
    class RunnableMeta(type):
        pass


else:
    RunnableMeta = type


class Runnable(metaclass=RunnableMeta):
    if t.TYPE_CHECKING:
        # this stub makes sure pyright accept so we want to check  constructor arguments.
        def __init__(*args, **kwargs):
            ...

        # this stub allows pyright to accept Runnable as Callables.
        def __call__(self, *args, **kwargs) -> t.Any:
            ...

    def __post__init__(self):
        object.__setattr__(self, '_state', _RunnableInternalState())
        self._state.is_initialized = True

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super(Runnable, cls).__init_subclass__(**kwargs)
        cls.__dict__.update((f"_{k}", v) for k, v in kwargs.items())
        cls._wrap_runnable_methods()
        cls._state = _uninitialized_runnable_state

        if not kwargs['batch'] and hasattr(cls, 'run_batch'):
            raise AttributeError(assert_msg.format(classname=self.__class__.__name__))

    @classmethod
    def _wrap_runnable_methods(cls):
        exc = [f.name for f in dc.fields(cls)] + [
            '__eq__',
            '__hash__',
            '__init__',
            '__repr__',
            '__post_init__',
        ]
        for key in _get_local_methods_name(cls, exclude=exc):
            _method = getattr(cls, key)
            setattr(cls, key, wrap_method_once(_method))

    def _call_wrapped_method(self, func, args, kw):
        is_setup_method = func.__name__ == 'setup'
        is_recurrent = self._state.in_setup
        if is_setup_method:
            self._state.in_setup = True
        else:
            self._try_setup()
        try:
            return func(self, *args, **kw)
        finally:
            if is_setup_method and not is_recurrent:
                self._state.reset()

    def _try_setup(self):
        if not self._state.setup_called and not self._state.in_setup:
            try:
                self._state.in_setup = True
                self.setup()
            finally:
                self._state.in_setup = False
                self._state.setup_called = True
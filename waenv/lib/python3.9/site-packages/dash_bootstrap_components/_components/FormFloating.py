# AUTO GENERATED FILE - DO NOT EDIT

import typing  # noqa: F401
from typing_extensions import TypedDict, NotRequired, Literal # noqa: F401
from dash.development.base_component import Component, _explicitize_args

ComponentType = typing.Union[
    str,
    int,
    float,
    Component,
    None,
    typing.Sequence[typing.Union[str, int, float, Component, None]],
]

NumberType = typing.Union[
    typing.SupportsFloat, typing.SupportsInt, typing.SupportsComplex
]


class FormFloating(Component):
    """A FormFloating component.
A component for adding float labels to form controls in forms.

Keyword arguments:

- children (a list of or a singular dash component, string or number; optional):
    The children of this FormFloating.

- id (string; optional):
    The ID of the FormFloating.

- html_for (string; optional):
    Set the `for` attribute of the label to bind it to a particular
    element.

- class_name (string; optional):
    Additional CSS classes to apply to the FormFloating.

- key (string; optional):
    A unique identifier for the component, used to improve performance
    by React.js while rendering components  See
    https://react.dev/learn/rendering-lists#why-does-react-need-keys
    for more info.

- className (string; optional):
    **DEPRECATED** Use `class_name` instead.  Additional CSS classes
    to apply to the FormFloating."""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_bootstrap_components'
    _type = 'FormFloating'


    def __init__(
        self,
        children: typing.Optional[ComponentType] = None,
        id: typing.Optional[typing.Union[str, dict]] = None,
        *,
        html_for: typing.Optional[str] = None,
        style: typing.Optional[typing.Any] = None,
        class_name: typing.Optional[str] = None,
        key: typing.Optional[str] = None,
        className: typing.Optional[str] = None,
        **kwargs
    ):
        self._prop_names = ['children', 'id', 'html_for', 'style', 'class_name', 'key', 'className']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['children', 'id', 'html_for', 'style', 'class_name', 'key', 'className']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}

        super(FormFloating, self).__init__(children=children, **args)

setattr(FormFloating, "__init__", _explicitize_args(FormFloating.__init__))

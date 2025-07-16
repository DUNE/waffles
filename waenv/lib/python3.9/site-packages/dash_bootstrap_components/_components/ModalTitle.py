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


class ModalTitle(Component):
    """A ModalTitle component.
Add a title to any Modal. Should be used as a child of the ModalHeader.

Keyword arguments:

- children (a list of or a singular dash component, string or number; optional):
    The children of this ModalTitle.

- id (string; optional):
    The ID of the ModalTitle.

- class_name (string; optional):
    Additional CSS classes to apply to the ModalTitle.

- tag (string; optional):
    HTML tag to use for the ModalTitle, default: div.

- className (string; optional):
    **DEPRECATED** Use `class_name` instead.  Additional CSS classes
    to apply to the ModalTitle."""
    _children_props = []
    _base_nodes = ['children']
    _namespace = 'dash_bootstrap_components'
    _type = 'ModalTitle'


    def __init__(
        self,
        children: typing.Optional[ComponentType] = None,
        id: typing.Optional[typing.Union[str, dict]] = None,
        *,
        style: typing.Optional[typing.Any] = None,
        class_name: typing.Optional[str] = None,
        tag: typing.Optional[str] = None,
        className: typing.Optional[str] = None,
        **kwargs
    ):
        self._prop_names = ['children', 'id', 'style', 'class_name', 'tag', 'className']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['children', 'id', 'style', 'class_name', 'tag', 'className']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}

        super(ModalTitle, self).__init__(children=children, **args)

setattr(ModalTitle, "__init__", _explicitize_args(ModalTitle.__init__))

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


class AccordionItem(Component):
    """An AccordionItem component.
A component to build up the children of the accordion.

Keyword arguments:

- id (string; optional):
    The ID of the AccordionItem.

- children (a list of or a singular dash component, string or number; optional):
    The children of the AccordionItem.

- title (a list of or a singular dash component, string or number; optional):
    Text to display in the header of the AccordionItem.

- item_id (string; optional):
    Optional identifier for item used for determining which item is
    visible if not specified, and AccordionItem is being used inside
    Accordion component, the item_id will be set to \"item-i\" where i
    is (zero indexed) position of item in list items passed to
    Accordion component.

- class_name (string; optional):
    Additional CSS classes to apply to the AccordionItem.

- className (string; optional):
    **DEPRECATED** Use `class_name` instead.  Additional CSS classes
    to apply to the AccordionItem."""
    _children_props = ['title']
    _base_nodes = ['title', 'children']
    _namespace = 'dash_bootstrap_components'
    _type = 'AccordionItem'


    def __init__(
        self,
        children: typing.Optional[ComponentType] = None,
        id: typing.Optional[typing.Union[str, dict]] = None,
        *,
        title: typing.Optional[ComponentType] = None,
        item_id: typing.Optional[str] = None,
        style: typing.Optional[typing.Any] = None,
        class_name: typing.Optional[str] = None,
        className: typing.Optional[str] = None,
        **kwargs
    ):
        self._prop_names = ['id', 'children', 'title', 'item_id', 'style', 'class_name', 'className']
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'children', 'title', 'item_id', 'style', 'class_name', 'className']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs and excess named props
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}

        super(AccordionItem, self).__init__(children=children, **args)

setattr(AccordionItem, "__init__", _explicitize_args(AccordionItem.__init__))

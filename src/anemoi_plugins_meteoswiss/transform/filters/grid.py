import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.grids import grid_registry


class AssignGrid(Filter):
    """A filter to assign a grid to the fields in a FieldList.

    This is a workaround for the lack of proper support for unstructured grids in
    earthkit-data. The `new_field_from_grid` function is used to add a callback to the Field
    object that will get the coordinates from the grid file when needed.
    """

    def __init__(self, grid_definition: dict):
        """Initialize the filter.

        Parameters
        ----------
        grid_definition:
            The grid definition dictionary.
        """
        self.grid_definition = grid_definition
        self.grid = grid_registry.from_config(self.grid_definition)
        super().__init__()

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        return new_fieldlist_from_list(
            [new_field_from_grid(field, self.grid) for field in data]
        )

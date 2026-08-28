Classes for reading and processing COMSOL Multiphysics VTU exports.
Powerful for post-processing of multiple exports with Python.

## Installation

Recommended via uv:
``uv add git+http://10.0.1.2:3000/tsimader/comsol_module.git --branch master``

To install dependencies:

`uv sync`


Currently, the module is able to read transient and stationary studies as well as sweeps from transient studies.

## Create Docs

```
uv sync --dev
cd docs
make html
```


## Examples

### Get array as point data
```[python]
    Example:
        >>> from comsol_module import ComsolVtu
        >>> vtu = ComsolVtu.from_file("simulation.vtu")
        >>> vtu.info()
        Dataset: simulation.vtu
        Path: simulation.vtu
        Study Type: Time-dependent
        Timesteps: 40 (from 0.000e+00 to 3.000e+13)
        Mesh Bounds: (-7.415000000000242e-06, 249999.99999999978, -1.9414062500000573e-06, 250000.00000000036, -7500.000000000005, 84.3298467930108)
        Points: 81782, Cells: 301702
        Available Fields:
        1: Pressure
        2: Temperature
        >>> T = vtu.get_array("Temperature")
        >>> T.shape
        (40, 81782)

```

### Get array as cell data

```[python]
    >>> vtu = ComsolVtu.from_file("simulation.vtu")
    >>> vtu.convert_to_cell_data()
    >>> T = vtu.get_array("Temperature", location = "cell")
    >>> T.shape
    (40, 301702)

```

### Plot data above a certain threshold
Use the underlying pyvista/vtk mesh object stored in `ComsolVtu`


```[python]
    >>> field_name = vtu.format_field("Temperature", -1)
    >>> field_name
    'Temperature_@_t=3E13'
    >>> filtered_mesh = vtu.mesh.threshold(scalars=field_name, value=480) # temperature above 480 K
    >>> filtered_mesh.plot(scalars=field_name)
```

### Inspect meta data
```[python]
    >>> vtu.time_values
    >>> vtu.time_keys
    >>> vtu.exported_fields
    >>> vtu.sweep_combos
```

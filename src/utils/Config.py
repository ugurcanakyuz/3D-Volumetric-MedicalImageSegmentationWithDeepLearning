from types import SimpleNamespace
import yaml


class NestedNamespace(SimpleNamespace):
    """A class that extends `SimpleNamespace` to support nested attribute assignment using dictionaries.

    This class allows you to create a nested namespace by providing a dictionary. Each key in the dictionary
    becomes an attribute of the object, and if the corresponding value is also a dictionary, it is converted
    into a nested `NestedNamespace` object.

    Notes
    -----
    Implemented from https://stackoverflow.com/a/54332748/13080899.

    Parameters
    ----------
    dictionary : dict
        The dictionary containing the attributes and their values for the namespace.
    **kwargs
        Additional keyword arguments to pass to the base class `SimpleNamespace`.

    Examples
    --------
    dictionary = {'a': 1, 'b': {'c': 2}}
    ns = NestedNamespace(dictionary)
    print(ns.a)  # Output: 1
    print(ns.b.c)  # Output: 2
    """

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


class Config(NestedNamespace):
    """A configuration class that inherits from `NestedNamespace`.

    This class is designed to load configuration data from a YAML file and create a nested namespace
    object containing the configuration values. It provides an easy way to access configuration options
    using dot notation.

    Parameters
    ----------
    path : str, optional
        The path to the YAML configuration file. Default is "config.yaml".
    **kwargs
        Additional keyword arguments to pass to the base class `NestedNamespace`.

    Examples
    --------
    Assuming a YAML file named "config.yaml" with the following content:
    ```
    a: 1
    b:
        c: 2
    ```
    You can create a `Config` object as follows:
    ```
    config = Config()
    print(config.a)  # Output: 1
    print(config.b.c)  # Output: 2
    """

    def __init__(self, path="config.yaml", **kwargs):
        """Load configuration data from a YAML file and create a nested namespace.

        Parameters
        ----------
        path : str, optional
            The path to the YAML configuration file. Default is "config.yaml".
        **kwargs
            Additional keyword arguments to pass to the base class `NestedNamespace`.
        """
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        super().__init__(config)

    def as_dict(self):
        """Convert the configuration object to a dictionary."""

        def convert_to_dict(obj):
            if isinstance(obj, NestedNamespace):
                return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
            else:
                return obj

        return convert_to_dict(self)
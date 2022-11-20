from types import SimpleNamespace
import yaml


class NestedNamespace(SimpleNamespace):
    # Implemented from https://stackoverflow.com/a/54332748/13080899 .
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


class Config(NestedNamespace):
    def __init__(self, path="config.yaml", **kwargs):
        with open(path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        super().__init__(config)
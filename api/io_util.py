from pathlib import Path
from typing import Union
import yaml, io


def load_config(path: Union[str, Path]):
    if not isinstance(path, str):
        path = str(path)

    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def load_multipart_file(file_content):
    file_content = file_content.file.read()
    file_content = file_content.decode()
    file_content = io.StringIO(file_content)
    return file_content



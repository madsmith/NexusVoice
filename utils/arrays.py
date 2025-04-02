import numpy as np


def abbreviate_data(data):
    """
    Return a string representation of the data:
      - Full list if <= 8 items
      - Otherwise: first 4, '... N ...', last 4
    Supports list, bytearray, or numpy array.
    """
    if isinstance(data, (bytearray, bytes)):
        data = list(data)
    elif isinstance(data, np.ndarray):
        data = data.tolist()
    elif not isinstance(data, list):
        raise TypeError(f"Unsupported type: {type(data)}")

    length = len(data)
    if length <= 8:
        return str(data)
    else:
        prefix = data[:4]
        suffix = data[-4:]
        middle_count = length - 8
        parts = prefix + [f"... {middle_count} ..."] + suffix
        return "[" + ", ".join(str(x) for x in parts) + "]"
from enum import Enum

# Command prefixes
COMMAND_QUERY_PREFIX = "?"
COMMAND_EXECUTE_PREFIX = "#"
COMMAND_RESPONSE_PREFIX = "~"

LINE_END = "\r\n"

class LutronSpecialEvents(Enum):
    AllEvents = "::[*]::"
    NonResponseEvents = "::[msg]::"
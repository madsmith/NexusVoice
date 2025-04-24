tool_registry = {}

def register_tool(tool_name: str):
    """
    Decorator to register a tool function.
    :param tool_name: The name of the tool.
    """
    def decorator(func):
        tool_registry[tool_name] = func
        return func
    return decorator
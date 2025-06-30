import asyncio
from fastmcp import Client
import mcp.types
import logfire
from logging import basicConfig

logfire.configure(service_name="MCP Client", environment="DEV")

basicConfig(handlers=[logfire.LogfireLoggingHandler()])

@logfire.instrument("MCP Client Main")
async def main():
    # Connect via stdio to a local script
    async with Client("nexusvoice/tools/lutron/server.py") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # results = await client.call_tool("getAreas", {})
        # assert isinstance(results[0], mcp.types.TextContent)
        # content = results[0]
        # print(f"Result: {content.text}")

        # results = await client.call_tool("getAreas", {"include_parents": True})
        # assert isinstance(results[0], mcp.types.TextContent)
        # content = results[0]
        # print(f"Result: {content.text}")

        results = await client.call_tool("getOutputs", {})
        assert isinstance(results[0], mcp.types.TextContent)
        content = results[0]
        print(f"Result: {content.text}")

        # import random
        # level = random.randint(0, 100)
        # results = await client.call_tool("setLevelForArea", {"area_iid": 26, "level": level})
        # if results:
        #     print(f"Result: {type(results)} {results}")
        


    # # Connect via SSE
    # async with Client("http://localhost:8000/sse") as client:
    #     # ... use the client
    #     pass

if __name__ == "__main__":
    asyncio.run(main())
    
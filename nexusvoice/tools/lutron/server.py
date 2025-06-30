import argparse
import asyncio
from http.client import NON_AUTHORITATIVE_INFORMATION
from fastmcp import FastMCP
import sys
from typing import Dict, List, Union, Optional

import logfire
import os
from logging import basicConfig
import logging.handlers
from nexusvoice.tools.lutron.lutron import LutronHomeworksClient
from nexusvoice.tools.lutron.commands import *
from nexusvoice.tools.lutron.database.database import LutronDatabase
from nexusvoice.tools.lutron.database.view import LutronDatabaseView, LutronArea, LutronAreaGroup, LutronOutput
from nexusvoice.core.config import load_config

DEFAULT_PORT = 8021
SERVER_NAME = "Lutron MCP Server"

logfire.configure(service_name=SERVER_NAME, environment="DEV")

# Configure additional file logging
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "server.log")

# Create file handler that logs to server.log
file_handler = logging.handlers.RotatingFileHandler(
    filename=log_file,
    maxBytes=10*1024*1024,  # 10MB file size
    backupCount=5,          # Keep 5 backup files
    encoding='utf-8',
)

# Set formatter for the file handler
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handler to the root logger
root_logger = logging.getLogger('')
root_logger.addHandler(file_handler)

class LutronMCPTools:
    def __init__(self, api: LutronHomeworksClient, dbView: LutronDatabaseView):
        self.api = api
        self.dbView = dbView

    def registerMCP(self, server: FastMCP):
        server.tool(self.getAreas)
        server.tool(self.getOutputs)
        server.tool(self.setLevelForArea)
    
    def getAreas(self, include_parents: bool = False) -> Dict[str, List[Union[LutronArea, LutronAreaGroup]]]:
        """
        Return all areas in the house.  Each area has a name and an 
        Integration ID (iid) upon which it can be referenced for area
        control actions.

        Args:
            include_parents: If True, include AreaGroup objects that
                contain other areas.  AreaGroup objects do not have an
                Integration ID (iid) but can be accessed by name.

        Returns:
            A dictionary containing two lists:
                areas: A list of LutronArea objects
                areaGroups: A list of LutronAreaGroup objects
        """

        areas = self.dbView.getAreas(include_parents)
        results = {
            'areas': [area for area in areas if isinstance(area, LutronArea)],
            'areaGroups': [area for area in areas if isinstance(area, LutronAreaGroup)]
        }
        return results

    async def setLevelForArea(self, area_iid: int, level: int):
        """
        Set the lighting level for a specific area.  An area is 
        typically a room or zone.  Each area has an Integration ID (iid)
        upon which it can be referenced for area control actions. The 
        level is a percentage (0-100).  With 0 meaning off and 100 
        meaning fully on.  For lights, a value of 75 is considered
        comfortably lit.

        Args:
            area_iid: The Integration ID of the area
            level: The level to set (0-100)
        """
        cmd = AreaCommand.set_level(area_iid, level)
        await cmd.execute(self.api)

    async def setLevelForAreaGroup(self, area_group_name: str, level: int):
        """
        Set the level for a specific area group.

        Args:
            area_group_name: The name of the area group
            level: The level to set (0-100)
        """
        pass

    def getOutputs(self):
        """
        Return all outputs in the house. Outputs are grouped by type of
        home component.  Each output has a name and an Integration ID
        (iid) upon which it can be referenced for output control actions.
        Each output will also have a path string that can be used to
        identify the location of the output in the house.

        Returns:
            A list of potential Output objects grouped by lights and
            shades.
        """
        outputs = self.dbView.getOutputs()
        result: dict[str, list[dict[str, int | str]]] = {}

        for output in outputs:
            if output.output_type not in result:
                result[output.output_type] = []
            result[output.output_type].append({
                "iid": output.iid,
                "name": output.name,
                "path": output.path
            })

        return result

    async def setLevelForOutput(self, output_iid: int, level: int):
        """
        Set the level for a specific output.  The level is a percentage
        (0-100).  With 0 meaning off/fully closed and 100 meaning fully
        on/fully open.  For lights, a value of 75 is considered
        comfortably lit.

        Args:
            output_iid: The Integration ID of the output
            level: The level to set (0-100)
        """
        cmd = OutputCommand.set_zone_level(output_iid, level)
        await cmd.execute(self.api)
    
async def run_server(args):
    config = load_config()
    server = FastMCP(SERVER_NAME)

    lutron_api = None

    with logfire.span("Lutron API Connect"):
        lutron_api = LutronHomeworksClient(
            config.get("tools.lutron.host"),
            config.get("tools.lutron.username"),
            config.get("tools.lutron.password")
        )
        await lutron_api.connect()
        await asyncio.sleep(1)
        logfire.info("Lutron API Connected")

    with logfire.span("Lutron Database Connect"):
        lutron_server = config.get("tools.lutron.host")
        database_path = config.get("tools.lutron.database_path")
        database = LutronDatabase(lutron_server, database_path)
        database.loadDatabase()

    with logfire.span("Lutron Database View Initialize"):
        dbView = LutronDatabaseView(config, database)
        dbView.initialize()

    tools = LutronMCPTools(lutron_api,dbView)
    tools.registerMCP(server)

    if args.mode == 'stdio':
        transport_kwargs = {}
    else:
        transport_kwargs = {
            "host": args.host,
            "port": args.port,
        }
    
    with logfire.span("MCP Server Run"):
        await server.run_async(
            transport=args.mode,
            **transport_kwargs,
        )
    
@logfire.instrument("MCP Server Main")
def main():
    parser = argparse.ArgumentParser(description="Lutron MCP Server")
    parser.add_argument("--mode", choices=["stdio", "sse", "http", "streamable-http"], default="stdio", help="Server transport mode (default: stdio)")
    sse_arg_group = parser.add_argument_group('SSE Server Mode Options')
    sse_arg_group.add_argument("--host", default="0.0.0.0", help="IP address to bind to (default: 0.0.0.0)")
    sse_arg_group.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for server to listen on (default: {DEFAULT_PORT})")
    
    args = parser.parse_args()
    
    asyncio.run(run_server(args))
    
if __name__ == "__main__":
    main()

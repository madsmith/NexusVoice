#!/usr/bin/env python3
"""
Simple telnet test client for Lutron controller.
Connects to host, reads data blocks, sends 'default' commands,
and saves all received bytes to session.raw.
"""

import asyncio
import logging
import os

from nexusvoice.core.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

config = load_config()

HOST = config.get('tools.lutron.host')
PORT = int(config.get('tools.lutron.port'))
CHUNK_SIZE = 1024
OUTPUT_FILE = './session.hex'


async def test_lutron_telnet():
    """Connect to the Lutron controller and test basic Telnet communication."""
    # Open file for binary writing
    with open(OUTPUT_FILE, 'wb') as output_file:
        logger.info(f"Connecting to {HOST}:{PORT}...")
        
        # Connect
        try:
            reader, writer = await asyncio.open_connection(HOST, PORT)
            logger.info(f"Connected to {HOST}:{PORT}")
            
            # First read - up to 1024 bytes
            logger.info("Reading first chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")
            
            # Send "default"
            logger.info("Sending 'default'...")
            writer.write(b"default\r\n")
            await writer.drain()
            
            # Second read - up to 1024 bytes
            logger.info("Reading second chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")
            
            # Send "default" again
            logger.info("Sending 'default' again...")
            writer.write(b"default\r\n")
            await writer.drain()
            
            # Third read - up to 1024 bytes
            logger.info("Reading third chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")


            # Send "?SYSTEM,1" again
            logger.info("Sending '?SYSTEM,1'...")
            writer.write(b"?SYSTEM,1\r\n")
            await writer.drain()
            
            # Fourth read - up to 1024 bytes
            logger.info("Reading fourth chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")


            # Send "?SYSTEM,2" again
            logger.info("Sending '?SYSTEM,2'...")
            writer.write(b"?SYSTEM,2\r\n")
            await writer.drain()
            
            # Fifth read - up to 1024 bytes
            logger.info("Reading fifth chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")


            # Sixth read - up to 1024 bytes
            logger.info("Reading sixth chunk...")
            data = await reader.read(CHUNK_SIZE)
            logger.info(f"Received {len(data)} bytes")
            output_file.write(data)
            
            # Hex dump for debugging
            logger.debug(f"Hex dump: {data.hex(' ')}")
            
        except Exception as e:
            logger.error(f"Error during Telnet session: {e}")
        finally:
            # Close the connection
            if 'writer' in locals():
                logger.info("Closing connection...")
                writer.close()
                await writer.wait_closed()
            
    logger.info(f"All received data saved to {OUTPUT_FILE}")
    

if __name__ == "__main__":
    asyncio.run(test_lutron_telnet())

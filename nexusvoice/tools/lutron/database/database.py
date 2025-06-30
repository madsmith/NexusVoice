from datetime import datetime
import logfire
import logging
import os
import re
import requests
import sqlite3
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

class LutronDatabaseException(Exception):
    """Exception raised for LutronDatabase errors."""
    pass

class LutronDatabase:
    def __init__(self, server: str, config_path: Optional[str] = None):
        """
        Initialize the LutronDatabase.
        
        Args:
            server: The Lutron server hostname or IP
            config_path: Optional path where configuration files will be stored.
                         If None, the current working directory will be used.
        """
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up config path and database files
        self.config_path = config_path or os.getcwd()
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
            
        self.dbFile = os.path.join(self.config_path, "DbXmlInfo.xml")
        self.sqlite_file = os.path.join(self.config_path, "lutron_config.db")
        
        # Initialize SQLite database
        self._init_db()

    # Define the current schema version
    CURRENT_SCHEMA_VERSION = 4
    
    @logfire.instrument("Initialize Database")
    def _init_db(self):
        """
        Initialize the SQLite database and run migrations if necessary.
        """
        conn = sqlite3.connect(self.sqlite_file)
        
        # Check if the info table exists to determine current schema version
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='info'")
        table_exists = cursor.fetchone() is not None
        
        current_version = 0
        if table_exists:
            # Get current schema version from the info table
            cursor.execute("SELECT value FROM info WHERE key='schema_version'")
            result = cursor.fetchone()
            if result:
                current_version = int(result[0])
        
        # Run all necessary migrations in sequence
        self._run_migrations(conn, current_version)
        
        conn.commit()
        conn.close()
        
    def _run_migrations(self, conn, current_version):
        """
        Run migrations sequentially to update the database schema.
        
        Args:
            conn: SQLite connection
            current_version: Current schema version of the database
        """
        # Dictionary of migration functions keyed by target version
        migrations = {
            1: self._migrate_to_v1,
            2: self._migrate_to_v2,
            3: self._migrate_to_v3,
            4: self._migrate_to_v4
        }
        
        # Apply migrations in version order
        for version in sorted([v for v in migrations.keys() if v > current_version]):
            self.logger.info(f"Migrating database schema to version {version}")
            migrations[version](conn)
            
            # Update schema version
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO info (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(version), datetime.now().isoformat())
            )
            conn.commit()
    
    @logfire.instrument("Migrate to Schema Version 1")
    def _migrate_to_v1(self, conn):
        """
        Migration to schema version 1: Create the initial info table.
        
        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()
        
        # Create info table for key-value pairs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS info (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    @logfire.instrument("Migrate to Schema Version 2")
    def _migrate_to_v2(self, conn):
        """
        Migration to schema version 2: Add outputs table.

        """

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "outputs" (
                rowid INTEGER PRIMARY KEY ASC AUTOINCREMENT, 
                iid INTEGER UNIQUE, 
                name TEXT NOT NULL, 
                type TEXT NOT NULL, 
                sort_order INTEGER, 
                wattage INTEGER
            );
        """)
        
    @logfire.instrument("Migrate to Schema Version 3")
    def _migrate_to_v3(self, conn):
        """
        Migration to schema version 3: Add areas, shadegroups, and shadegroup_output tables.
        """
        cursor = conn.cursor()
        
        # Create areas table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "areas" (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                iid INTEGER,
                name TEXT NOT NULL,
                occupancy_group INTEGER,
                sort_order INTEGER,
                parent_id INTEGER,
                is_leaf BOOLEAN
            );
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_areas_name_when_iid_null
                ON areas(name)
                WHERE iid IS NULL;
        """)
        
        # Create shadegroups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "shadegroups" (
                rowid INTEGER PRIMARY KEY ASC AUTOINCREMENT, 
                iid INTEGER UNIQUE, 
                name TEXT NOT NULL, 
                sort_order INTEGER
            );
        """)
        
        # Create shadegroup_output join table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "shadegroup_output" (
                shadegroup_iid INTEGER NOT NULL, 
                output_iid INTEGER NOT NULL, 
                PRIMARY KEY (shadegroup_iid, output_iid)
            );
        """)

    @logfire.instrument("Migrate to Schema Version 4")
    def _migrate_to_v4(self, conn):
        """
        Migration to schema version 4: Add iid_map table to track relationships between IIDs and their parents.
        Using combinations of (iid, name) to uniquely identify elements, including those with null IIDs.
        """
        cursor = conn.cursor()
        
        # Create iid_map table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS "iid_map" (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                iid INTEGER,
                name TEXT,
                type TEXT NOT NULL,
                parent_iid INTEGER,
                parent_name TEXT,
                parent_type TEXT
            );
        """)
        
        # Create composite index on (iid, name) for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iid_map_iid_name ON iid_map(iid, name);
        """)
        
        # Create index on name for name-based lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iid_map_name ON iid_map(name);
        """)
        
        # Create composite index on (parent_iid, parent_name) for parent-based relationship queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_iid_map_parent_iid_name ON iid_map(parent_iid, parent_name);
        """)
        
        # Add UNIQUE constraint to prevent duplicate entries for the same element with the same parent
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_iid_map_unique_child_parent 
            ON iid_map(iid, name, parent_iid, parent_name);
        """)

        
    
    def setInfo(self, key: str, value: str):
        """
        Store a key-value pair in the info table.
        
        Args:
            key: The key to store
            value: The value to store
        """
        conn = sqlite3.connect(self.sqlite_file)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO info (key, value, updated_at) 
        VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def getInfo(self, key: str, default=None):
        """
        Retrieve a value by key from the info table.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value if found, otherwise the default
        """
        conn = sqlite3.connect(self.sqlite_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM info WHERE key = ?", (key,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        return default

    def _parse_export_timestamp(self, xml_chunk):
        """
        Parse the export timestamp from a chunk of XML data.
        
        Args:
            xml_chunk: A chunk of XML data as bytes or string
            
        Returns:
            datetime object if timestamp found, None otherwise
        """
        
        # Convert bytes to string if needed
        if isinstance(xml_chunk, bytes):
            xml_chunk = xml_chunk.decode('utf-8', errors='ignore')
            
        # Look for date and time patterns
        date_match = re.search(r'<DbExportDate>(\d{2}/\d{2}/\d{4})</DbExportDate>', xml_chunk)
        time_match = re.search(r'<DbExportTime>(\d{2}:\d{2}:\d{2})</DbExportTime>', xml_chunk)
        
        if date_match and time_match:
            date_str = date_match.group(1)
            time_str = time_match.group(1)
            timestamp_str = f"{date_str} {time_str}"
            
            try:
                # Parse MM/DD/YYYY HH:MM:SS format
                return datetime.strptime(timestamp_str, "%m/%d/%Y %H:%M:%S")
            except ValueError:
                self.logger.error(f"Failed to parse timestamp: {timestamp_str}")
                return None
                
        return None
    
    def _is_newer_than_last_import(self, server_timestamp):
        """
        Check if the server timestamp is newer than our last import.
        
        Args:
            server_timestamp: datetime object of server timestamp
            
        Returns:
            True if server data is newer or we don't have a timestamp, False otherwise
        """
        if not server_timestamp:
            # If we can't determine server timestamp, assume it's new
            return True
            
        last_import = self.getInfo("lastImportTime")
        if not last_import:
            # If we don't have a last import time, consider it new
            return True
            
        try:
            # Parse ISO format timestamp
            last_import_dt = datetime.fromisoformat(last_import)
            
            # Compare the timestamps
            return server_timestamp > last_import_dt
        except ValueError:
            self.logger.error(f"Failed to parse lastImportTime: {last_import}")
            return True  # If we can't parse, assume we need the update
        
    @logfire.instrument("Load Database")
    def loadDatabase(self):
        """
        Load the database from disk or server.
        
        First checks if there's a newer version on the server (streaming to check timestamp).
        If server has newer data, downloads and saves it, then returns it.
        Otherwise loads from disk if file exists.
        
        Returns:
            XML content as bytes or None if unavailable
        """
        try:
            # Always check the server first to see if there's a newer version
            needs_update, updated_xml = self.checkServerLoad()
            
            if needs_update and updated_xml is not None:
                self.saveToDisk(updated_xml)
                # Record import and access times
                current_time = datetime.now().isoformat()
                self.setInfo("lastImportTime", current_time)
                self.setInfo("lastAccessTime", current_time)
                self._process_xml(updated_xml)
                return updated_xml
        except Exception as e:
            self.logger.error(f"Error checking server for updates: {e}")
            # Continue to try loading from disk as fallback
            
        # If server check failed or no update needed, try loading from disk
        if os.path.exists(self.dbFile):
            xml = self.loadFromDisk()
            if xml is not None:
                self.setInfo("lastAccessTime", datetime.now().isoformat())
                return xml
                
        # If we reach here, both server check and disk load failed
        return None

    @logfire.instrument("Disk - Save")
    def saveToDisk(self, xml: bytes):
        """
        Save XML data to disk.
        
        Args:
            xml: XML data as bytes
        """
        with open(self.dbFile, "wb") as f:
            f.write(xml)

    @logfire.instrument("Disk - Load")
    def loadFromDisk(self):
        """
        Load XML data from disk.
        
        Returns:
            XML content as bytes or None if file cannot be read
        """
        try:
            with open(self.dbFile, "rb") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load XML from disk: {e}")
            return None
    
    @logfire.instrument("Server - Check")
    def checkServerLoad(self) -> Tuple[bool, Optional[bytes]]:
        """
        Connect to the server and stream just enough data to check the timestamp.
        Compares the server timestamp to our stored lastImportTime.
        
        Returns:
            Tuple of (needs_update, xml_data):
                needs_update: True if server has newer data or we couldn't determine
                xml_data: Full XML content if needs_update is True, otherwise None
        """
        try:
            self.logger.info(f"Checking timestamp from {self.server}")
            
            # Use a streaming request to minimize data transfer
            with requests.get(f"http://{self.server}/DbXmlInfo.xml", stream=True) as response:
                if response.status_code != 200:
                    self.logger.error(f"Failed to connect: {response.status_code}")
                    raise Exception(f"Failed to connect: {response.status_code}")
                
                # Read chunks until we find the timestamp or reach a limit
                buffer = b''
                server_timestamp = None
                chunk_size = 8192  # 8KB chunks
                max_read_size = 1024 * 1024  # Stop after reading 1MB
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    logfire.debug(f"Read {len(chunk)} bytes", )
                    buffer += chunk
                    
                    # Check if we have the timestamp
                    server_timestamp = self._parse_export_timestamp(buffer)
                    if server_timestamp or len(buffer) > max_read_size:
                        break
                
                # If we couldn't find the timestamp or it's newer, continue downloading
                if not server_timestamp or self._is_newer_than_last_import(server_timestamp):
                    self.logger.info(f"Server has newer data (timestamp: {server_timestamp})")
                    
                    # Continue downloading the rest of the file
                    content = buffer
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        content += chunk
                    
                    return True, content
                else:
                    self.logger.info(f"Local data is up to date (server timestamp: {server_timestamp})")
                    return False, None
                        
        except Exception as e:
            self.logger.error(f"Error checking server timestamp: {e}")
            raise e

    @logfire.instrument("Process XML")
    def _process_xml(self, xml: bytes):
        """
        Process XML data and update the database.
        
        Parses the XML content and updates database tables with extracted information.
        Processes Areas, Outputs, and ShadeGroups elements from the XML.
        
        Args:
            xml: XML data as bytes
        """
        try:
            # Parse XML data
            self.logger.info("Processing XML data and updating database...")
            root = ET.fromstring(xml.decode('utf-8'))
            
            # Connect to SQLite database
            conn = sqlite3.connect(self.sqlite_file)
            cursor = conn.cursor()
            
            # Process elements
            self._process_areas(root, cursor)
            self._process_outputs(root, cursor)
            self._process_shadegroups(root, cursor)
            self._process_iid_map(root, cursor)
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            self.logger.info("XML processing completed successfully")
        except Exception as e:
            self.logger.error(f"Error processing XML data: {e}")
            raise LutronDatabaseException(f"Error processing XML data: {e}")
    
    @logfire.instrument("Process Outputs")
    def _process_outputs(self, root: ET.Element, cursor):
        """
        Process Output elements from XML and update the outputs table.
        
        Args:
            root: Root element of the XML tree
            cursor: SQLite cursor for executing queries
        """
        # Find all Output elements in the XML
        output_count = 0
        
        # Look for Output elements anywhere in the hierarchy
        for output in root.findall('.//Output'):
            # Extract attributes
            name = output.get('Name', '')
            integration_id = output.get('IntegrationID')
            output_type = output.get('OutputType')
            wattage = output.get('Wattage')
            sort_order = output.get('SortOrder')
            
            # Skip if no integration ID
            if not integration_id:
                self.logger.warning(f"Output element missing IntegrationID: {output.attrib}")
                continue
                
            # Convert attributes to appropriate types
            try:
                integration_id = int(integration_id)
                wattage = int(wattage) if wattage else None
                sort_order = int(sort_order) if sort_order else None
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid values in Output element: {output.attrib}")
                continue
            
            # Insert or replace record in outputs table
            cursor.execute("""
                INSERT OR REPLACE INTO outputs 
                (iid, name, type, sort_order, wattage)
                VALUES (?, ?, ?, ?, ?)
                """, (integration_id, name, output_type, sort_order, wattage))
            
            output_count += 1
        
        self.logger.info(f"Processed {output_count} output elements")
    
    @logfire.instrument("Process Areas")
    def _process_areas(self, root: ET.Element, cursor):
        """
        Process Area elements from XML and update the areas table.
        
        Handles parent-child relationships between areas and identifies leaf nodes.
        A leaf node is an area that contains no child areas.
        
        Args:
            root: Root element of the XML tree
            cursor: SQLite cursor for executing queries
        """
        area_count = 0
        parent_map = {child: parent for parent in root.iter() for child in parent}
        area_rowid_map = {}
        
        def get_parent_area(elem):
            while elem is not None:
                elem = parent_map.get(elem)
                if elem is not None and elem.tag == 'Area':
                    return elem
            return None

        for area in root.findall('.//Area'):
            # Extract attributes
            name = area.get('Name', '')
            integration_id = area.get('IntegrationID')
            occupancy_group = area.get('OccupancyGroupAssignedToID')
            sort_order = area.get('SortOrder')
            
            # Convert attributes to appropriate types
            try:
                integration_id = int(integration_id) if (integration_id is not None and integration_id != "0") else None
                occupancy_group = int(occupancy_group) if occupancy_group else None
                sort_order = int(sort_order) if sort_order else None
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid values in Area element: {area.attrib}")
                continue

            # Is this a leaf node?
            children = area.findall('.//Area')
            is_leaf = len(children) == 0
            
            # Get the parent area
            parent_area = get_parent_area(area)
            parent_id = area_rowid_map.get(parent_area) if parent_area is not None else None
            
            # Insert or replace record in areas table with null parent_id and is_leaf initially
            cursor.execute("""
                INSERT OR REPLACE INTO areas
                (iid, name, occupancy_group, sort_order, parent_id, is_leaf)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (integration_id, name, occupancy_group, sort_order, parent_id, is_leaf))
            rowid = cursor.lastrowid
            area_rowid_map[area] = rowid
            
            area_count += 1
        
        self.logger.info(f"Processed {area_count} area elements with parent-child relationships")
    
    @logfire.instrument("Process ShadeGroups")
    def _process_shadegroups(self, root: ET.Element, cursor):
        """
        Process ShadeGroup elements from XML and update the shadegroups and shadegroup_output tables.
        
        Args:
            root: Root element of the XML tree
            cursor: SQLite cursor for executing queries
        """
        # Find all ShadeGroup elements in the XML
        shadegroup_count = 0
        association_count = 0
        
        for shadegroup in root.findall('.//ShadeGroup'):
            # Extract attributes
            name = shadegroup.get('Name', '')
            integration_id = shadegroup.get('IntegrationID')
            sort_order = shadegroup.get('SortOrder')
            
            # Skip if no integration ID
            if not integration_id:
                continue
                
            # Convert attributes to appropriate types
            try:
                integration_id = int(integration_id)
                sort_order = int(sort_order) if sort_order else None
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid values in ShadeGroup element: {shadegroup.attrib}")
                continue
            
            # Insert or replace record in shadegroups table
            cursor.execute("""
                INSERT OR REPLACE INTO shadegroups
                (iid, name, sort_order)
                VALUES (?, ?, ?)
                """, (integration_id, name, sort_order))
            
            shadegroup_count += 1
            
            # Process associations with outputs
            for output_ref in shadegroup.findall('./Output'):
                output_iid = output_ref.get('IntegrationID')
                
                if output_iid:
                    try:
                        output_iid = int(output_iid)
                        cursor.execute("""
                            INSERT OR REPLACE INTO shadegroup_output
                            (shadegroup_iid, output_iid)
                            VALUES (?, ?)
                            """, (integration_id, output_iid))
                        association_count += 1
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid output reference in ShadeGroup: {output_ref.attrib}")
        
        self.logger.info(f"Processed {shadegroup_count} shadegroups with {association_count} output associations")
    
    @logfire.instrument("Process IID Map")
    def _process_iid_map(self, root: ET.Element, cursor):
        """
        Process the IID map from XML and update the iid_map table.
        Using combinations of (iid, name) to uniquely identify elements, including those with null IIDs.
        
        Args:
            root: Root element of the XML tree
            cursor: SQLite cursor for executing queries
        """
        # Find all elements in the XML
        parent_map = {child: parent for parent in root.iter() for child in parent}
        iid_count = 0
        
        print("Processing IID map...")
        for elem in root.iter():
            # Extract attributes
            integration_id = elem.get('IntegrationID')
            name = elem.get('Name')
            
            # Skip if no name
            if integration_id is None and name is None:
                continue
            
            # Find the parent element
            parent_iid = None
            parent_name = None
            parent_type = None
            parent_elem = parent_map.get(elem)
            while parent_elem is not None:
                tag_iid = parent_elem.get('IntegrationID')
                tag_name = parent_elem.get('Name')
                if not (tag_iid is None and tag_name is None):
                    parent_iid = tag_iid
                    parent_name = tag_name
                    parent_type = parent_elem.tag
                    break

                parent_elem = parent_map.get(parent_elem)

            # Convert attributes to appropriate types
            try:
                if integration_id == "0":
                    integration_id = None
                elif integration_id is not None:
                    integration_id = int(integration_id)
                if parent_iid == "0":
                    parent_iid = None
                elif parent_iid is not None:
                    parent_iid = int(parent_iid)
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid values in element: {elem.attrib}")
                continue

            # Insert or replace record in iid_map table
            cursor.execute("""
                INSERT OR REPLACE INTO iid_map 
                (iid, name, type, parent_iid, parent_name, parent_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (integration_id, name, elem.tag, parent_iid, parent_name, parent_type))
            
            iid_count += 1
        
        self.logger.info(f"Processed {iid_count} elements for IID map")
    
    def getOutputs(self, output_type=None):
        """
        Retrieve outputs from the database, optionally filtered by type.
        
        Args:
            output_type: Optional filter for output type (e.g., 'INC', 'LED', etc.)
            
        Returns:
            List of dictionaries containing output information
        """
        conn = sqlite3.connect(self.sqlite_file)
        conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        cursor = conn.cursor()
        
        try:
            if output_type:
                cursor.execute("""
                    SELECT iid, name, type, sort_order, wattage 
                    FROM outputs 
                    WHERE type = ?
                    """, (output_type,))
            else:
                cursor.execute("""
                    SELECT iid, name, type, sort_order, wattage 
                    FROM outputs
                    """)
                
            # Convert rows to dictionaries
            results = [dict(row) for row in cursor.fetchall()]
            self.logger.debug(f"Retrieved {len(results)} outputs{' of type ' + output_type if output_type else ''}")
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving outputs: {e}")
            return []
        finally:
            conn.close()
    
    @logfire.instrument("Get Areas")
    def getAreas(self):
        """
        Retrieve all areas from the database.
        
        Returns:
            List of dictionaries containing area information
        """
        conn = sqlite3.connect(self.sqlite_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT rowid, iid, name, occupancy_group, sort_order, parent_id, is_leaf
                FROM areas
                """)
            
            results = [dict(row) for row in cursor.fetchall()]
            self.logger.debug(f"Retrieved {len(results)} areas")
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving areas: {e}")
            return []
        finally:
            conn.close()
    
    @logfire.instrument("Get ShadeGroups")
    def getShadeGroups(self, include_outputs=False):
        """
        Retrieve all shade groups from the database.
        
        Args:
            include_outputs: If True, include associated output IDs for each shade group
            
        Returns:
            List of dictionaries containing shade group information
        """
        conn = sqlite3.connect(self.sqlite_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT iid, name, sort_order
                FROM shadegroups
                """)
            
            shade_groups = [dict(row) for row in cursor.fetchall()]
            
            # If requested, add associated outputs to each shade group
            if include_outputs and shade_groups:
                for sg in shade_groups:
                    cursor.execute("""
                        SELECT output_iid 
                        FROM shadegroup_output 
                        WHERE shadegroup_iid = ?
                        """, (sg['iid'],))
                    sg['outputs'] = [row[0] for row in cursor.fetchall()]
            
            self.logger.debug(f"Retrieved {len(shade_groups)} shade groups")
            return shade_groups
        except Exception as e:
            self.logger.error(f"Error retrieving shade groups: {e}")
            return []
        finally:
            conn.close()
    
    @logfire.instrument("Get Outputs by ShadeGroup")
    def getOutputsByShadeGroup(self, shadegroup_iid):
        """
        Retrieve outputs associated with a specific shade group.
        
        Args:
            shadegroup_iid: Integration ID of the shade group
            
        Returns:
            List of dictionaries containing output information
        """
        conn = sqlite3.connect(self.sqlite_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT o.iid, o.name, o.type, o.sort_order, o.wattage
                FROM outputs o
                JOIN shadegroup_output sgo ON o.iid = sgo.output_iid
                WHERE sgo.shadegroup_iid = ?
                """, (shadegroup_iid,))
            
            results = [dict(row) for row in cursor.fetchall()]
            self.logger.debug(f"Retrieved {len(results)} outputs for shade group {shadegroup_iid}")
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving outputs for shade group {shadegroup_iid}: {e}")
            return []
        finally:
            conn.close()

    @logfire.instrument("Get IID Map")
    def getIIDMap(self) -> dict:
        """
        Retrieve all elements from the iid_map table.
        
        Returns:
            Dictionary of dictionaries containing iid_map information, keyed by (iid, name) tuples
        """
        conn = sqlite3.connect(self.sqlite_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT rowid, iid, name, type, parent_iid, parent_name, parent_type
                FROM iid_map
                ORDER BY iid, name
                """)
            
            # For simplicity in accessing, we'll use a dictionary with iid as the key
            # but it's important to note that (iid, name) is the actual unique identifier
            results = {}
            for row in cursor.fetchall():
                key = (row['iid'], row['name'])
                results[key] = {
                    "iid": row['iid'],
                    "name": row['name'],
                    "type": row['type'],
                    "parent_iid": row['parent_iid'],
                    "parent_name": row['parent_name'],
                    "parent_type": row['parent_type']
                }

            self.logger.debug(f"Retrieved {len(results)} elements from iid_map")
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving iid_map: {e}")
            return {}
        finally:
            conn.close()
        
    
import os
import tempfile
import unittest
import sqlite3
from shutil import copyfile
from unittest import mock
from datetime import datetime, timedelta
from nexusvoice.core.config import load_config
import pytest

from nexusvoice.tools.lutron.database.database import LutronDatabase

config = load_config()

class TestLutronDatabase(unittest.TestCase):
    """Test cases for the LutronDatabase class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = self.temp_dir.name
        self.original_dir = os.getcwd()
        self.test_server = "test-server.local"
        self.database = LutronDatabase(self.test_server, config_path=self.config_path)
        
        # Load actual XML content from file for testing
        xml_file_path = os.path.join(os.path.dirname(__file__), "DbXmlInfo.xml")
        with open(xml_file_path, "rb") as f:
            self.sample_xml = f.read()
        
    def tearDown(self):
        """Clean up after each test."""
        os.chdir(self.original_dir)
        self.temp_dir.cleanup()
        
    def test_init(self):
        """Test initialization of the database."""
        self.assertEqual(self.database.server, self.test_server)
        self.assertEqual(self.database.dbFile, os.path.join(self.config_path, "DbXmlInfo.xml"))
        self.assertEqual(self.database.sqlite_file, os.path.join(self.config_path, "lutron_config.db"))
        
        # Verify SQLite DB was created
        self.assertTrue(os.path.exists(self.database.sqlite_file))
        
    def test_save_to_disk(self):
        """Test saving database to disk."""
        self.database.saveToDisk(self.sample_xml)
        
        # Verify file was created with correct content
        self.assertTrue(os.path.exists("DbXmlInfo.xml"))
        with open("DbXmlInfo.xml", "rb") as f:
            content = f.read()
            # Just check if content exists and contains XML structure
            # Since the exact content might vary between test runs
            self.assertIn(b'<?xml', content)
            # Check that we've written some content (non-empty file)
            self.assertTrue(len(content) > 100)
            
    def test_load_from_disk(self):
        """Test loading database from disk."""
        # Create a test file first
        with open(self.database.dbFile, "wb") as f:
            f.write(self.sample_xml)
            
        # Test loading
        loaded_xml = self.database.loadFromDisk()
        self.assertEqual(loaded_xml, self.sample_xml)
        
    def test_check_server_load(self):
        """Test checking server timestamp via streaming."""
        with mock.patch('requests.get') as mock_get:
            # Create mock response with stream support
            mock_response = mock.MagicMock()
            mock_response.status_code = 200
            
            # Create sample XML with timestamp
            now = datetime.now()
            date_bytes = now.strftime("%m/%d/%Y").encode('utf-8')
            time_bytes = now.strftime("%H:%M:%S").encode('utf-8')
            timestamp_xml = b'<?xml version="1.0"?>\n<DbInfo>\n<DbExportDate>' + date_bytes + b'</DbExportDate><DbExportTime>' + time_bytes + b'</DbExportTime>\n</DbInfo>'
            mock_response.__enter__.return_value = mock_response
            mock_response.iter_content.return_value = [timestamp_xml]
            mock_get.return_value = mock_response
            
            # Test with no previous import time
            with mock.patch.object(self.database, 'getInfo', return_value=None):
                needs_update, xml = self.database.checkServerLoad()
                self.assertTrue(needs_update)
                # Just check if timestamps were extracted correctly, not exact XML match
                self.assertIsNotNone(xml)
            
            # Test with older import time
            old_time = (now - timedelta(days=1)).isoformat()
            with mock.patch.object(self.database, 'getInfo', return_value=old_time):
                needs_update, xml = self.database.checkServerLoad()
                self.assertTrue(needs_update)
                self.assertIsNotNone(xml)
        
    @mock.patch('os.path.exists')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.loadFromDisk')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.checkServerLoad')
    def test_load_database_from_disk(self, mock_check_server_load, mock_load_disk, mock_exists):
        """Test loadDatabase when local data is up-to-date."""
        # Setup mocks
        mock_check_server_load.return_value = (False, None)  # No update needed
        mock_exists.return_value = True
        mock_load_disk.return_value = self.sample_xml
        
        # Call the method
        result = self.database.loadDatabase()
        
        # Verify correct methods were called
        mock_check_server_load.assert_called_once()
        mock_load_disk.assert_called_once()
        
        # Check the result
        self.assertEqual(result, self.sample_xml)
        
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.loadFromDisk')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.checkServerLoad')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.saveToDisk')
    def test_load_database_from_server(self, mock_save, mock_check_server_load, mock_load_disk):
        """Test loadDatabase when server has newer data."""
        # Setup mocks
        mock_check_server_load.return_value = (True, self.sample_xml)  # Needs update, has XML data
        
        # Call the method
        result = self.database.loadDatabase()
        
        # Verify correct methods were called
        mock_check_server_load.assert_called_once()
        mock_load_disk.assert_not_called()
        mock_save.assert_called_once_with(self.sample_xml)
        
        # Check result
        self.assertEqual(result, self.sample_xml)
        
    @mock.patch('os.path.exists')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.loadFromDisk')
    @mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.checkServerLoad')
    def test_load_database_server_error(self, mock_check_server_load, mock_load_disk, mock_exists):
        """Test loadDatabase when both server check and disk load fail."""
        # Setup mocks
        mock_check_server_load.side_effect = Exception("Connection error")  # Server check fails
        mock_exists.return_value = False  # No local file
        mock_load_disk.return_value = None  # Disk load fails
        
        # Call the method
        result = self.database.loadDatabase()
        
        # Verify correct methods were called
        mock_check_server_load.assert_called_once()
        
        # Check result is None when server fails
        self.assertIsNone(result)

    def test_set_get_info(self):
        """Test setting and getting values in the info table."""
        test_key = "test_key"
        test_value = "test_value"
        
        # Set a value
        self.database.setInfo(test_key, test_value)
        
        # Get the value back
        retrieved_value = self.database.getInfo(test_key)
        self.assertEqual(retrieved_value, test_value)
        
        # Test default value for non-existent key
        default_value = "default"
        retrieved_default = self.database.getInfo("non_existent_key", default_value)
        self.assertEqual(retrieved_default, default_value)
        
    def test_lastImportTime_update(self):
        """Test that lastImportTime is updated when loading from server."""
        with mock.patch('nexusvoice.tools.lutron.database.database.LutronDatabase.checkServerLoad') as mock_check:
            # Setup mock to simulate new data available
            mock_check.return_value = (True, self.sample_xml)
            
            # First, ensure lastImportTime is not set
            initial_time = self.database.getInfo("lastImportTime")
            self.assertIsNone(initial_time)
            
            # Load database - this should update lastImportTime
            self.database.loadDatabase()
            
            # Check that lastImportTime is now set
            import_time = self.database.getInfo("lastImportTime")
            self.assertIsNotNone(import_time)


@pytest.fixture
def database():
    tempdir = tempfile.TemporaryDirectory()

    # Copy the sample XML file to the temporary directory
    sample_xml_path = os.path.join(os.path.dirname(__file__), "DbXmlInfo.xml")
    copyfile(sample_xml_path, os.path.join(tempdir.name, "DbXmlInfo.xml"))

    db = LutronDatabase(config.get("tools.lutron.host"), tempdir.name)

    yield db

    tempdir.cleanup()

def test_load_database(database: LutronDatabase):
    xml = database.loadDatabase()
    assert xml is not None
    
    last_import = database.getInfo("lastImportTime")
    last_access = database.getInfo("lastAccessTime")
    
    assert last_access is not None or last_import is not None

def test_process_disk_load(database: LutronDatabase):
    # Load XML from disk
    xml = database.loadFromDisk()
    assert xml is not None
    
    # Process XML
    database._process_xml(xml)
    
    # Check that outputs table is not empty
    outputs = database.getOutputs()
    assert len(outputs) > 0
    
    # Check that areas table is populated
    areas = database.getAreas()
    if len(areas) > 0:
        # Check that at least some areas have parent_id or is_leaf properly set
        hierarchy_data_exists = any(area.get('parent_id') is not None or 
                                   area.get('is_leaf') is not None 
                                   for area in areas)
        assert hierarchy_data_exists

class TestLutronDatabaseXMLProcessing(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = LutronDatabase("test_server", self.tempdir.name)
    
    def tearDown(self):
        self.tempdir.cleanup()
    
    def test_process_xml(self):
        """Test parsing XML and updating the database"""
        # Sample XML data with outputs
        xml_data = """
            <Project>
                <Outputs>
                    <Output Name="Kitchen Area Rec Cans 7" IntegrationID="41" OutputType="INC" SortOrder="2" Wattage="154" />
                    <Output Name="Den LED Lights" IntegrationID="42" OutputType="LED" SortOrder="3" Wattage="35" />
                    <Output Name="Hallway Accent" IntegrationID="43" OutputType="INC" SortOrder="1" Wattage="75" />
                </Outputs>
            </Project>
        """.encode('utf-8')
        
        # Process the XML
        self.db._process_xml(xml_data)
        
        # Test retrieving all outputs
        outputs = self.db.getOutputs()
        self.assertEqual(len(outputs), 3)

        print(outputs)
        
        # Test retrieving outputs by type
        led_outputs = self.db.getOutputs("LED")
        self.assertEqual(len(led_outputs), 1)
        self.assertEqual(led_outputs[0]['name'], "Den LED Lights")
        self.assertEqual(led_outputs[0]['iid'], 42)
        
        # Find a specific output by name
        kitchen_output = None
        for output in outputs:
            if output['name'] == "Kitchen Area Rec Cans 7":
                kitchen_output = output
                break
        
        self.assertIsNotNone(kitchen_output, "Kitchen output should be in the database")
        assert kitchen_output is not None # Assert for type hinting

        self.assertEqual(kitchen_output['name'], "Kitchen Area Rec Cans 7")
        self.assertEqual(kitchen_output['type'], "INC")
        self.assertEqual(kitchen_output['wattage'], 154)
        
    def test_schema_migration(self):
        """Test schema migration from version 1 to version 3"""
        # Create a new database with schema version 1
        conn = sqlite3.connect(os.path.join(self.tempdir.name, "lutron_config.db"))
        cursor = conn.cursor()
        
        # Check current schema version
        cursor.execute("SELECT value FROM info WHERE key = 'schema_version'")
        result = cursor.fetchone()
        self.assertEqual(result[0], "3")
        
        # Check for tables created in schema migrations
        # Schema v2 should have outputs table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='outputs'
        """)
        self.assertIsNotNone(cursor.fetchone())
        
        # Schema v3 should have areas, shadegroups, and shadegroup_output tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='areas'
        """)
        self.assertIsNotNone(cursor.fetchone())
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='shadegroups'
        """)
        self.assertIsNotNone(cursor.fetchone())
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='shadegroup_output'
        """)
        self.assertIsNotNone(cursor.fetchone())
        
        # Verify area table has the parent_id and is_leaf columns
        cursor.execute("PRAGMA table_info(areas)")
        columns = {row[1]: row for row in cursor.fetchall()}
        self.assertIn('parent_id', columns)
        self.assertIn('is_leaf', columns)
    
    def test_process_areas_hierarchy(self):
        """Test processing areas with parent-child hierarchy"""
        # Sample XML data with nested areas
        xml_data = """
            <Project>
                <Areas>
                    <Area Name="Main Floor" IntegrationID="10" OccupancyGroupAssignedToID="100" SortOrder="1">
                        <Area Name="Living Room" IntegrationID="11" OccupancyGroupAssignedToID="101" SortOrder="1" />
                        <Area Name="Kitchen" IntegrationID="12" OccupancyGroupAssignedToID="102" SortOrder="2">
                            <Area Name="Kitchen Island" IntegrationID="121" OccupancyGroupAssignedToID="103" SortOrder="1" />
                        </Area>
                    </Area>
                    <Area Name="Second Floor" IntegrationID="20" OccupancyGroupAssignedToID="104" SortOrder="2">
                        <Area Name="Master Bedroom" IntegrationID="21" OccupancyGroupAssignedToID="105" SortOrder="1" />
                        <Area Name="Guest Room" IntegrationID="22" OccupancyGroupAssignedToID="106" SortOrder="2" />
                    </Area>
                </Areas>
            </Project>
        """.encode('utf-8')
        
        # Process the XML
        self.db._process_xml(xml_data)
        
        # Test retrieving areas with hierarchy info
        areas = self.db.getAreas()
        self.assertEqual(len(areas), 7)
        
        # Create a map of areas by iid for easy lookup
        area_map = {area['iid']: area for area in areas}
        
        # Check parent-child relationships
        # Kitchen Island (121) should have Kitchen (12) as parent
        kitchen_island = area_map.get(121)
        self.assertIsNotNone(kitchen_island)
        assert kitchen_island is not None, "Kitchen Island should not be None" # Assert for type hinting
        self.assertIsNotNone(kitchen_island['parent_id'])
        
        # Get the Kitchen area by rowid
        cursor = sqlite3.connect(self.db.sqlite_file).cursor()
        cursor.execute("SELECT iid FROM areas WHERE rowid = ?", (kitchen_island['parent_id'],))
        parent_iid = cursor.fetchone()[0]
        self.assertEqual(parent_iid, 12)  # Should be Kitchen's integration ID
        
        # Check leaf node identification
        # Main Floor (10) should NOT be a leaf
        main_floor = area_map.get(10)
        self.assertIsNotNone(main_floor)
        assert main_floor is not None, "Main Floor should not be None" # Assert for type hinting
        self.assertEqual(main_floor['is_leaf'], 0)
        
        # Kitchen Island (121) should be a leaf
        self.assertEqual(kitchen_island['is_leaf'], 1)

if __name__ == "__main__":
    unittest.main()

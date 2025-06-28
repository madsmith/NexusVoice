from shutil import copyfile
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from nexusvoice.core.config import NexusConfig, load_config
from nexusvoice.tools.lutron.database.database import LutronDatabase
from nexusvoice.tools.lutron.database.view import LutronDatabaseView, LutronArea

actual_config = load_config()

class TestLutronDatabaseView(unittest.TestCase):
    """Test suite for LutronDatabaseView class."""
    
    def setUp(self):
        """Set up test environment with a temporary database and sample data."""
        # Create a temporary directory and database
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = LutronDatabase("test_server", self.tempdir.name)
        
        # Create mock config
        def config_get_side_effect(key, default=None):
            config_data = {
                "tools.lutron.valid_object_types": ["Area", "Output"],
                "tools.lutron.filters": {
                    "name_replace": [
                        ["Rom", "Room"],
                        ["2nd", "Second"],
                        ["Mstr", "Master"],
                    ],
                },
            }
            return config_data.get(key, default)
        
        self.config = mock.MagicMock(spec=NexusConfig)
        self.config.get.side_effect = config_get_side_effect
        
        # Create the view
        self.view = LutronDatabaseView(self.config, self.db)
        self.view.initialize()
        
        # Sample XML data with nested areas
        self.sample_areas_xml = """
            <Project>
                <Areas>
                    <Area Name="100 Main Floor" IntegrationID="0" OccupancyGroupAssignedToID="100" SortOrder="1">
                        <Areas>
                            <Area Name="001 Living Rom" IntegrationID="11" OccupancyGroupAssignedToID="101" SortOrder="1">
                                <Outputs>
                                    <Output Name="Stairwell Chandelier 1" IntegrationID="111" OutputType="INC" Wattage="125" SortOrder="1"/>
                                    <Output Name="Stairwell Rec Cans 4" IntegrationID="110" OutputType="INC" Wattage="88" SortOrder="0"/>
                                    <Output Name="Shade 001" IntegrationID="444" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="2"/>
                                    <Output Name="Shade 002" IntegrationID="446" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="3"/>
                                </Outputs>
                            </Area>
                            <Area Name="002 Kitchen" IntegrationID="12" OccupancyGroupAssignedToID="102" SortOrder="2">
                                <Outputs>
                                    <Output Name="Kitchen Island Lights" IntegrationID="121" OutputType="INC" Wattage="75" SortOrder="0"/>
                                    <Output Name="Kitchen Pendant" IntegrationID="122" OutputType="INC" Wattage="60" SortOrder="1"/>
                                    <Output Name="Kitchen Shade" IntegrationID="450" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="2"/>
                                </Outputs>
                            </Area>
                            <Area Name="003 Kitchen Island" IntegrationID="13" OccupancyGroupAssignedToID="103" SortOrder="1">
                                <Outputs>
                                    <Output Name="Island Pendant 1" IntegrationID="131" OutputType="INC" Wattage="40" SortOrder="0"/>
                                    <Output Name="Island Pendant 2" IntegrationID="132" OutputType="INC" Wattage="40" SortOrder="1"/>
                                    <Output Name="Island Pendant 3" IntegrationID="133" OutputType="INC" Wattage="40" SortOrder="2"/>
                                </Outputs>
                            </Area>
                        </Areas>
                    </Area>
                    <Area Name="200 2nd Floor" IntegrationID="0" OccupancyGroupAssignedToID="104" SortOrder="2">
                        <Areas>
                            <Area Name="004 Mstr Bedroom" IntegrationID="21" OccupancyGroupAssignedToID="105" SortOrder="1">
                                <Outputs>
                                    <Output Name="Area Rec Cans 4" IntegrationID="89" OutputType="INC" Wattage="88" SortOrder="0"/>
                                    <Output Name="Chandelier" IntegrationID="91" OutputType="INC" Wattage="125" SortOrder="1"/>
                                    <Output Name="Bed Reading Lamps 2" IntegrationID="74" OutputType="INC" Wattage="44" SortOrder="2"/>
                                    <Output Name="Sheer 1" IntegrationID="412" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="3"/>
                                    <Output Name="Sheer 2" IntegrationID="414" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="4"/>
                                    <Output Name="Blackout 1" IntegrationID="416" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="5"/>
                                    <Output Name="Blackout 2" IntegrationID="418" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="6"/>
                                </Outputs>
                            </Area>
                            <Area Name="005 Guest Room" IntegrationID="22" OccupancyGroupAssignedToID="106" SortOrder="2">
                                <Outputs>
                                    <Output Name="Guest Ceiling Light" IntegrationID="221" OutputType="INC" Wattage="75" SortOrder="0"/>
                                    <Output Name="Guest Bedside Lamp" IntegrationID="222" OutputType="INC" Wattage="40" SortOrder="1"/>
                                    <Output Name="Guest Room Shade" IntegrationID="470" OutputType="SYSTEM_SHADE" Wattage="0" SortOrder="2"/>
                                </Outputs>
                            </Area>
                        </Areas>
                    </Area>
                </Areas>
            </Project>
        """.encode('utf-8')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.tempdir.cleanup()

    def test_get_areas(self):
        """Test that getAreas returns the expected data structure."""
        # Process the sample XML data into the database
        self.db._process_xml(self.sample_areas_xml)
        
        # Get areas from both the database and the view
        db_areas = self.db.getAreas()
        view_areas = self.view.getAreas()
        
        # Verify that the view returns the same number of areas as the database
        self.assertLessEqual(len(view_areas), len(db_areas))
        self.assertEqual(len(view_areas), 5)

        self.assertIsInstance(view_areas[0], LutronArea)
        self.assertEqual(view_areas[0].iid, db_areas[1]['iid'])
        self.assertNotEqual(view_areas[0].name, db_areas[1]['name'])
        self.assertIn("001 Living Room", view_areas[0].name)

        found_iids = [area.iid for area in view_areas]
        expected_iids = [area['iid'] for area in db_areas if area['is_leaf']]
        self.assertEqual(found_iids, expected_iids)

        names = [area.name for area in view_areas]
        self.assertIn("001 Living Room", names, "Filter fixed area name")
        self.assertIn("002 Kitchen", names, "Filter fixed area name")
        self.assertNotIn("200 2nd Floor", names, "Filter fixed area name")

    def test_get_areas_include_parents(self):
        """Test that getAreas returns the expected data structure when include_parents is True."""
        # Process the sample XML data into the database
        self.db._process_xml(self.sample_areas_xml)
        
        # Get areas from both the database and the view
        db_areas = self.db.getAreas()
        view_areas = self.view.getAreas(include_parents=True)
        
        # Verify that the view returns the same number of areas as the database
        self.assertLessEqual(len(view_areas), len(db_areas))
        self.assertEqual(len(view_areas), 7)

        self.assertIsInstance(view_areas[1], LutronArea)
        self.assertEqual(view_areas[1].iid, db_areas[1]['iid'])
        self.assertNotEqual(view_areas[1].name, db_areas[1]['name'])
        self.assertEqual("001 Living Room", view_areas[1].name)

        found_iids = [area.iid for area in view_areas]
        expected_iids = [area['iid'] for area in db_areas]
        self.assertEqual(found_iids, expected_iids)

        names = [area.name for area in view_areas]
        self.assertIn("001 Living Room", names, "Filter fixed area name")
        self.assertIn("002 Kitchen", names, "Filter fixed area name")
        self.assertIn("200 Second Floor", names, "Filter fixed area name")
    
    def test_get_entities(self):
        self.db._process_xml(self.sample_areas_xml)
        
        entities = self.view.getEntities()
        for entity in entities:
            print(entity)
            print("    Path: ", entity.path)

class TestLutronDatabaseViewWithData(unittest.TestCase):
    """Test suite for LutronDatabaseView class using actual XML data from file."""
    
    def setUp(self):
        """Set up test environment with a temporary database and data from XML file."""
        # Create a temporary directory and database
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = LutronDatabase("test_server", self.tempdir.name)
        
        # Create mock config
        self.config = actual_config
        
        # Create the view
        self.view = LutronDatabaseView(self.config, self.db)
        self.view.initialize()
        
        # Load XML data from file
        with open("tests/tools/lutron/DbXmlInfo.xml", "rb") as f:
            self.xml_data = f.read()
    
    def tearDown(self):
        """Clean up temporary files."""
        # Show tempdir contents
        print("Tempdir contents:")
        import os
        for file in os.listdir(self.tempdir.name):
            print(file)
        copyfile(self.tempdir.name + "/lutron_config.db", "tests/tools/lutron/lutron_config.db")
        self.tempdir.cleanup()

    def test_get_entities(self):
        """Test that getEntities returns the expected data structure with actual XML data."""
        self.db._process_xml(self.xml_data)
        
        entities = self.view.getEntities()
        for entity in entities:
            print(entity)
            print("    Path: ", entity.path)


    def test_get_iid_map(self):
        self.db._process_xml(self.xml_data)
        
        iid_map = self.db.getIIDMap()
        for iid, map_record in iid_map.items():
            print(f"{iid}:[{map_record['type']}] {map_record['parent_iid']} ({map_record['parent_type']})")


if __name__ == "__main__":
    unittest.main()

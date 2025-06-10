import sys
import os
from nexusvoice.bootstrap import bootstrap

# Determine if the executing script is in the 'scripts' folder
script_path = os.path.abspath(sys.argv[0])
service_name = None
scripts_path_prefix = os.path.sep + 'scripts' + os.path.sep
if scripts_path_prefix in script_path or script_path.endswith(scripts_path_prefix):
    environment_mode = 'DEV'
    service_name = os.path.basename(script_path)
elif 'TEST_RUN_PIPE' in os.environ:
    environment_mode = 'TEST'
    service_name = "pytest"
else:
    environment_mode = 'PROD'
    service_name = "NexusVoice"

bootstrap(environment_mode, service_name=service_name)
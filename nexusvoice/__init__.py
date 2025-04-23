import sys
import os
from nexusvoice.bootstrap import setup_environment, initialize_openwakeword

# Determine if the executing script is in the 'scripts' folder
script_path = os.path.abspath(sys.argv[0])
if os.path.sep + "scripts" + os.path.sep in script_path or script_path.endswith(os.path.sep + "scripts"):
    environment_mode = 'DEV'
else:
    environment_mode = 'PROD'

setup_environment(environment_mode)
initialize_openwakeword()
import sys
import os
from nexusvoice.bootstrap import bootstrap

# Determine if the executing script is in the 'scripts' folder
script_path = os.path.abspath(sys.argv[0])
if os.path.sep + "scripts" + os.path.sep in script_path or script_path.endswith(os.path.sep + "scripts"):
    environment_mode = 'DEV'
else:
    environment_mode = 'PROD'

#print(f"Bootstraping environment: {environment_mode}")
bootstrap(environment_mode)
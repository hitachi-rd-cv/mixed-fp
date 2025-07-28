import os
import re
import sys
import toml
from dotenv import load_dotenv

from lib_common.waluigi import pop_cmndline_arg
from tasks import *
from pipelines import *

if __name__ == '__main__':
    load_dotenv()
    os.environ['LUIGI_CONFIG_PARSER'] = 'toml'

    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    path_config = sys.argv.pop(2)
    if '-m' in sys.argv:
        conf_func_name = pop_cmndline_arg('-m')
    else:
        conf_func_name = 'get_config'

    exec(f'from {path_config} import {conf_func_name}')

    params = eval(f'{conf_func_name}()')

    # save params as toml with random name
    basename = path_config.split(".")[-1]
    os.makedirs('.tmp', exist_ok=True)
    with open(f'.tmp/{basename}.toml', 'w') as f:
        toml.dump(params, f)

    gokart.add_config(f'.tmp/{basename}.toml')
    gokart.run()
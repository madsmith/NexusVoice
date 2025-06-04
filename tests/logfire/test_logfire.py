from nexusvoice.bootstrap import bootstrap


import logfire
from datetime import datetime

def test_logfire():
    time = datetime.now()
    logfire.configure(token='pylf_v1_us_YKJqXzd1zQcRjzjDbmRsGHN2prwy9rBrfDNmZdTVgkjZ')
    logfire.info('Hello, {place}!', place=f'World at {time}')


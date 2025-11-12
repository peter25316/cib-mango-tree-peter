import sys
sys.path.insert(0, 'src')
from tests import main

print('Running data preparation (cached or fresh)...')
ana, ts = main.run_data_preparation(force_rerun=False)
print('Data prepared:', ana is not None)
print('Running burst detection...')
res = main.run_burst_detection(ana, force_rerun=False)
print('Burst detection returned types:', tuple(type(x) for x in res))
if res and len(res) >= 3 and res[2] is not None:
    print('Number of bursts with contributors:', len(res[2]))
    if len(res[2]) > 0:
        first = res[2][0]
        print('First burst summary:', {k: first.get(k) for k in ['start_time','end_time','post_count']})


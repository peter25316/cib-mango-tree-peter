import sys
sys.path.insert(0, 'src')
from tests import main

# Force fresh run to ensure mapping is up-to-date
analyzer, ts_df = main.run_data_preparation(force_rerun=True)
print('Data prepared:', analyzer is not None)

burst_list, posts_with_bursts, contributors = main.run_burst_detection(analyzer, force_rerun=True)
if burst_list is None:
    print('No bursts found or burst detection failed.')
    raise SystemExit(1)

print(f"Total bursts: {len(burst_list)}\n")

# For each burst, print summary and contributors
for i, b in enumerate(burst_list):
    level = b.get('level')
    start = b.get('start_time')
    end = b.get('end_time')
    print('---')
    print(f'Burst #{i} — level={level}  start={start}  end={end}')
    # contributors list should align
    if contributors and i < len(contributors):
        c = contributors[i]
        pc = c.get('post_count', 0)
        print(f'  post_count: {pc}')
        top_accounts = c.get('top_accounts') or []
        print('  top_accounts:')
        for ta in top_accounts[:10]:
            print('   ', ta)
        posts = c.get('posts') or []
        print(f'  sample posts (first {min(5,len(posts))}):')
        for p in posts[:5]:
            # robust access
            pid = p.get('id') if isinstance(p, dict) else None
            uname = p.get('account.username') or p.get('account.username') if isinstance(p, dict) else None
            disp = p.get('account.display_name') if isinstance(p, dict) else None
            content = p.get('content_cleaned') if isinstance(p, dict) else str(p)
            snippet = (content[:300] + '...') if content and len(content) > 300 else content
            print('    id:', pid, 'user:', uname, 'display:', disp)
            print('     ', snippet)
    else:
        print('  No contributor info for this burst')

print('\nDone. Full contributor objects are cached in cache/burst_results.pkl')


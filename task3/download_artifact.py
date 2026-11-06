import argparse
import concurrent.futures
import hashlib
import html
import re
import time
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--hash-prefix')
    p.add_argument('--index')
    args = p.parse_args()
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = args.hash_prefix
    if args.index:
        response = requests.get(args.index, timeout=(10,30))
        response.raise_for_status()
        links = [html.unescape(s) for s in re.findall(r'href="([^"]+)"', response.text)]
        candidates = [s for s in links if path.name.replace('+','%2B')+'#sha256=' in s]
        assert len(candidates) == 1, candidates
        expected = candidates[0].split('#sha256=')[1]
    assert expected, 'A trusted expected hash is required'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest().startswith(expected):
        print('ALREADY_VERIFIED',path,flush=True)
        return
    initial = requests.get(args.url, headers={'Range':'bytes=0-0'}, timeout=(10,30))
    assert initial.status_code == 206, (initial.status_code,initial.text[:100])
    size = int(initial.headers['Content-Range'].split('/')[1])
    temp = path.with_suffix(path.suffix+'.part')
    with temp.open('wb') as f:
        f.truncate(size)
    chunk = 1024**2
    jobs = [(s,min(size-1,s+chunk-1)) for s in range(0,size,chunk)]
    def part(bounds):
        start,end = bounds
        for attempt in range(4):
            try:
                r = requests.get(args.url,headers={'Range':f'bytes={start}-{end}'},timeout=(12,40))
                assert r.status_code == 206
                assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                assert len(r.content) == end-start+1
                with temp.open('r+b') as f:
                    f.seek(start)
                    f.write(r.content)
                return len(r.content)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)
    print('DOWNLOAD_START',path.name,round(size/1024**2,2),'MiB',flush=True)
    total=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for n in pool.map(part,jobs):
            total+=n
            if total%(8*chunk)==0 or total==size:
                print('DOWNLOAD',f'{total/size:.1%}',flush=True)
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    assert digest.startswith(expected), (digest,expected)
    temp.replace(path)
    print('SHA256_VERIFIED',digest,str(path),flush=True)


if __name__=='__main__':
    main()

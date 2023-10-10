import os
import gzip
import glob
import sys
import time
import datetime
import shutil
import requests
import contextlib
from tqdm import tqdm
from utils import Colorful
from multiprocessing import Pool


def reformat(timestamps):
    for ts in timestamps:
        for i in range(0, 800, 10):
            os.makedirs(f'CC/2023-23-{ts}-{i:05d}-{i + 9:05d}', exist_ok=True)
        new_dirs = [f'CC/2023-23-{ts}-{i:05d}-{i + 9:05d}' for i in range(0, 800, 10)]
        dirs = glob.glob(f'{ts}/*.warc.gz')
        dirs.sort(key=lambda x: int(x[-13:-8]))
        while dirs:
            tgt_dir = new_dirs.pop(0)
            for i in range(10):
                shutil.move(dirs.pop(0), tgt_dir)
        assert not os.listdir(ts), 'reformat may be failed'
        os.removedirs(ts)


def check_integrity(frag):
    with tqdm(desc=colorful.blue(frag), colour='GREEN', position=0, total=10) as bar:
        for file in glob.glob(f'CC/{frag}/*.warc.gz'):
            check_pass = False
            while not check_pass:
                times = 0
                try:
                    with gzip.open(file, 'rb') as f:
                        f.seek(-1, os.SEEK_END)
                        check_pass = True
                except Exception as e:
                    print(e, colorful.yellow(f'`{file}` is broken'), colorful.green(f'redownload `{times}` times tried'))
                    os.remove(file)
                    url = f'https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-23/segments/{body[frag[8:22]]}/warc/{file.split("/")[-1]}'
                    resp = requests.get(url, stream=True, timeout=10)
                    with contextlib.closing(resp) as r, open(file, 'wb') as g:
                        for seg in tqdm(r.iter_content(chunk_size=(2 ** 20) * 10), total=120):
                            g.write(seg)
                    times += 1
            bar.update(1)


body = {'20230527223515': '1685224643388.45',
        '20230528015553': '1685224643462.13',
        '20230528051321': '1685224643585.23',
        '20230528083025': '1685224643663.27',
        '20230528114832': '1685224643784.62',
        '20230528150639': '1685224644309.7'}


def g(t):
    with tqdm(position=0, total=t, dynamic_ncols=True) as bar:
        for _ in range(t):
            time.sleep(0.1)
            print(f'{datetime.datetime}')
            bar.update(1)


if __name__ == '__main__':
    colorful = Colorful()
    frags = os.listdir('CC')

    # reformat(timestamps=)
    with Pool(5) as pool:
        for _ in tqdm.auto(pool.imap(g, [5] * 5)):
            pass

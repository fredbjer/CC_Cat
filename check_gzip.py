import os
import json
import gzip
import glob
import shutil

import requests
import contextlib
from tqdm import tqdm
from utils import Colorful
from multiprocessing import Pool


def load_seg_map(path):
    if os.path.isfile(path):
        return json.load(open(path, 'r'))
    dic = {}
    with open('./warc.paths', 'r') as f:
        lines = f.readlines()
        for line in lines:
            seg = line.split('/')
            dic[seg[-1][8:22]] = seg[3]
    json.dump(dic, open('seg_map.json', 'w'), ensure_ascii=False, indent=2)
    return dic


def reformat(timestamps):
    for ts in timestamps:
        if not os.path.isdir(ts):
            continue
        for i in range(0, 800, 10):
            os.makedirs(f'CC-checked/2023-23-{ts}-{i:05d}-{i + 9:05d}', exist_ok=True)
        new_dirs = [f'CC-checked/2023-23-{ts}-{i:05d}-{i + 9:05d}' for i in range(0, 800, 10)]
        dirs = glob.glob(f'{ts}/*.warc.gz')
        dirs.sort(key=lambda x: int(x[-13:-8]))
        while dirs:
            tgt_dir = new_dirs.pop(0)
            for i in range(10):
                shutil.move(dirs.pop(0), tgt_dir)
        assert not os.listdir(ts), 'reformat may be failed'
        os.removedirs(ts)


def check_single(path):
    try:
        with gzip.open(path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
        return True
    except Exception as e:
        return e and False


def check_integrity(frag):
    with tqdm(desc=colorful.blue(frag), colour='GREEN', position=0, total=10) as bar:
        for file in sorted(glob.glob(f'CC-checked/{frag}/*.warc.gz')):
            check_pass = False
            tries = 0
            while not check_pass:
                try:
                    with gzip.open(file, 'rb') as f:
                        f.seek(-1, os.SEEK_END)
                        check_pass = True
                except Exception as e:
                    try:
                        tries += 1
                        tqdm.write(str(e) + ' ' + colorful.yellow(f'`{tries}` th trying redownload `{file}`'))
                        if os.path.isfile(file):
                            os.remove(file)
                        url = f'{body}/{seg_map[frag[8:22]]}/warc/{file.split("/")[-1]}'
                        resp = requests.get(url, stream=True, timeout=10)
                        with contextlib.closing(resp) as r, open(file, 'wb') as g:
                            for seg in tqdm(r.iter_content(chunk_size=(2 ** 20) * 10), desc=file.split('/')[-1], total=116):
                                g.write(seg)
                    except Exception as ei:
                        print(str(ei) + ' ' + colorful.red(f'Failed on `{file}`'))
            bar.update(1)


body = 'https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-23/segments'
seg_map = load_seg_map('seg_map.json')


if __name__ == '__main__':
    colorful = Colorful()

    reformat(timestamps=['20230527223515', '20230528015553'])

    frags = os.listdir('CC-checked')
    with Pool(len(frags)) as pool:
        _ = list(pool.imap_unordered(check_integrity, frags))

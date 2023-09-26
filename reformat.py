import os
import glob
import shutil

for i in range(0, 800, 10):
    os.makedirs(f'CC-new/2023-23-20230528015553-{i:05d}-{i + 9:05d}')
new_dirs = [f'CC-new/2023-23-20230528015553-{i:05d}-{i + 9:05d}' for i in range(0, 800, 10)]
dirs = glob.glob('CC-2/*.warc.gz')
dirs.sort(key=lambda x: int(x[-13:-8]))
while dirs:
    tgt_dir = new_dirs.pop(0)
    for i in range(10):
        shutil.move(dirs.pop(0), tgt_dir)

import os
import re
import json
import glob
import gzip
import pickle
import signal
import timeit
import datetime
import fasttext
import functools
# import traceback

from tqdm import tqdm
from tld import get_fld
from utils import Colorful
from trafilatura import extract
from multiprocessing import Pool
from collections import defaultdict, namedtuple
from warcio.archiveiterator import ArchiveIterator
# from fastwarc.warc import ArchiveIterator, WarcRecordType
# from fastwarc.stream_io import FileStream, GZipStream
from langdetect import detect, LangDetectException


class Counter:
    def __init__(self):
        self.pdf = 0
        self.xml = 0
        self.json = 0
        self.image = 0
        self.video = 0
        self.audio = 0
        self.octet_stream = 0

        self.total = 0
        self.normal = 0
        self.useful = 0
        self.unknown = 0

        self.domain_pass = 0
        self.adult_re_pass = 0
        self.encoding_pass = 0
        self.trafilatura_pass = 0
        self.lang_detection_pass = 0

        self.doc_len_fail = 0
        self.word_len_fail = 0
        self.token_kind_fail = 0
        self.token_rate_fail = 0
        self.unicode_rate_fail = 0
        self.paragraph_intersection_fail = 0

        self.language = defaultdict(int)
        self.start = timeit.default_timer()

    def __add__(self, other):
        counter = Counter()
        for attr, val in counter.__dict__.items():
            if isinstance(val, int):
                setattr(counter, attr, getattr(self, attr) + getattr(other, attr))
            counter.start = min(self.start, other.start)
        for k, v in self.language.items():
            counter.language[k] += v
        for k, v in other.language.items():
            counter.language[k] += v
        return counter

    def __enter__(self):
        return self

    def __report__(self):
        print(colorful.timer(self.start))
        print(colorful.blue(f'{"PDF files":^20}: `{self.pdf:>10,d}` || rate: `{self.pdf / self.total:.5f}`'))
        print(colorful.blue(f'{"XML files":^20}: `{self.xml:>10,d}` || rate: `{self.xml / self.total:.5f}`'))
        print(colorful.blue(f'{"JSON files":^20}: `{self.json:>10,d}` || rate: `{self.json / self.total:.5f}`'))
        print(colorful.blue(f'{"IMAGE files":^20}: `{self.image:>10,d}` || rate: `{self.image / self.total:.5f}`'))
        print(colorful.blue(f'{"VIDEO files":^20}: `{self.video:>10,d}` || rate: `{self.video / self.total:.5f}`'))
        print(colorful.blue(f'{"Octet-Stream files":^20}: `{self.octet_stream:>10,d}` || rate: `{self.octet_stream / self.total:.5f}`'))
        print(colorful.green(f'{"Total process":^20}: `{self.total:>10,d}`'))
        print(colorful.blue(f'{"domain pass":^20}: `{self.domain_pass:>10,d}` || rate: `{self.domain_pass / self.total:.5f}`'))
        print(colorful.blue(f'{"adult regex pass":^20}: `{self.adult_re_pass:>10,d}` || rate: `{self.adult_re_pass / self.total:.5f}`'))
        print(colorful.blue(f'{"content extract pass":^20}: `{self.trafilatura_pass:>10,d}` || rate: `{self.trafilatura_pass / self.total:.5f}`'))
        print(colorful.blue(f'{"lang detect pass":^20}: `{self.lang_detection_pass:>10,d}` || rate: `{self.lang_detection_pass / self.total:.5f}`'))
        black_hit = self.unknown + self.useful
        print(colorful.green(f'{"Normal domains":^20}: `{self.normal:>10,d}` || rate: `{self.normal / self.total:.5f}`'))
        print(colorful.blue(f'{"Total hit blacklist":^20}: `{black_hit:>10,d}` || rate: `{black_hit / self.total:.5f}`'))
        print(colorful.blue(f'{"hit `useful`":^20}: `{self.useful:>10,d}` || rate: `{self.useful / self.total:.5f}`'))
        print(colorful.blue(f'{"hit `unknown`":^20}: `{self.unknown:>10,d}` || rate: `{self.unknown / self.total:.5f}`'))
        print(colorful.blue(self.language.__str__()))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type and not exc_val and not exc_tb:
            if not hasattr(self, 'bar'):
                self.bar = tqdm
            if not hasattr(self, 'file'):
                self.file = ''
            self.bar.write(colorful.green(f'Successful finished `{self.file}` at `{datetime.datetime.now()}`'))


def load_blacklist(path):
    if os.path.isfile('blacklist.bin'):
        return pickle.load(open('blacklist.bin', 'rb'))
    blacklist = {}
    for file in tqdm(glob.glob(path, recursive=True)):
        mark, category = file[10:-8].split('/')
        with open(file, 'r') as f:
            for domain in f.read().splitlines():
                blacklist[domain] = dir_names(mark, category)
    if not os.path.isfile('blacklist.bin'):
        pickle.dump(blacklist, open('blacklist.bin', 'wb'))
    return blacklist


def quality_filter(extra, lang, counter, frag):
    def saver(fail_type, content):
        os.makedirs(path := f'extract/quality_filter/{fail_type}', exist_ok=True)
        with open(path + f'/{frag}.jsonl', 'a+') as f:
            f.write(json.dumps(content) + '\n')

    if lang not in non_space_separated:
        words = extra.split()
        if not 50 <= len(words) <= 100000:
            counter.doc_len_fail += 1
            return False
        if not 3 <= sum(len(s) for s in words) / len(words) <= 10:
            counter.word_len_fail += 1
            return False
        if len(set(words)) / len(words) < 0.2:
            counter.token_kind_fail += 1
            saver('token_kind_fail', {lang: extra})
            return False
        if lang == 'en' and sum(w in stopwords[lang] for w in words) / len(words) < 0.1:
            counter.token_rate_fail += 1
            saver('token_rate_fail', {lang: extra})
            return False
        if lang == 'en' and sum(all(0x0041 <= ord(s) <= 0x005a or 0x0061 <= ord(s) <= 0x007a for s in w) for w in words) / len(words) < 0.8:
            counter.unicode_rate_fail += 1
            saver('unicode_rate_fail', {lang: extra})
            return False
    else:
        if not 50 <= len(extra) <= 100000:
            counter.doc_len_fail += 1
            return False
        if len(set(extra)) / len(extra) < 0.2:
            counter.token_kind_fail += 1
            saver('token_kind_fail', {lang: extra})
            return False
        if lang in 'zh_ja' and sum(s in stopwords[lang] for s in extra) / len(extra) < 0.05:
            counter.token_rate_fail += 1
            saver('token_rate_fail', {lang: extra})
            return False
        if lang == 'zh' and sum(0x4e00 <= ord(s) <= 0x9fa5 for s in extra) / len(extra) < 0.8:
            counter.unicode_rate_fail += 1
            saver('unicode_rate_fail', {lang: extra})
            return False
        if lang == 'ja' and sum(0x4e00 <= ord(s) <= 0x9fa5 or 0x3040 <= ord(s) <= 0x309f or 0x30a0 <= ord(s) <= 0x30ff for s in extra) / len(extra) < 0.8:
            counter.unicode_rate_fail += 1
            saver('unicode_rate_fail', {lang: extra})
            return False

    paragraphs = extra.split('\n')
    if len(paragraphs) >= 2:
        intersection = set(paragraphs[0] if lang in non_space_separated else paragraphs[0].split())
        union = set()
        for para in paragraphs[1:]:
            para = set(para if lang in non_space_separated else para.split())
            intersection &= para
            union |= para
        if len(intersection) / len(union) >= 0.5:
            counter.paragraph_intersection_fail += 1
            saver('paragraph_intersection_fail', {lang: extra})
            return False
    return True


def writer(url, path):
    os.makedirs(path, exist_ok=True)
    # with open(path + f'/{name}', 'wb') as f:
    #     f.write(record)
    with open(path + f'/url_index.jsonl', 'a+') as f:
        f.write(json.dumps({'url': url}) + '\n')


def save_items(arrows, frag, counter):
    for path, records in arrows.items():
        os.makedirs(path, exist_ok=True)
        with open(path + f'/{frag}.jsonl', 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps({'data': record}, ensure_ascii=False) + '\n')

    os.makedirs(path := f'extract/Counter', exist_ok=True)
    with open(path + f'/{frag}', 'wb') as f:
        pickle.dump(counter, f)


def load_index(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def tracer(func):
    @functools.wraps(func)
    def wrapper(frag):
        try:
            func(frag)
        except Exception as e:
            print(e, colorful.red(f'subprocess exited on `{frag}`'))
            # for t in traceback.format_exception(e, limit=-1):
            #     print(colorful.green(t[:-2]))
            os.killpg(os.getpgid(os.getgid()), signal.SIGKILL)
    return wrapper


@tracer
def warc_extract(frag):
    arrows = defaultdict(set)
    # import fsspec
    # with fsspec.open('https://data.commoncrawl.org/' + frag, 'rb', compression='gzip') as stream
    with Counter() as counter, tqdm(desc=colorful.blue(frag), colour='GREEN', position=0) as bar:
        setattr(counter, 'bar', bar)
        for file in glob.glob(f'CC/{frag}/*.warc.gz'):
            setattr(counter, 'file', file)
            for record in ArchiveIterator(gzip.open(file, 'r')):
                if record.rec_type != 'response':
                    continue
                # for record in ArchiveIterator(GZipStream(FileStream(file, 'rb')), record_types=WarcRecordType.response):
                counter.total += 1

                url = record.rec_headers.get_header('WARC-Target-URI')
                # url = record.headers['WARC-Target-URI']
                if not (domain := get_fld(url, fix_protocol=True, fail_silently=True)):
                    continue
                if domain in black_list and black_list[domain].mark == 'strict':
                    continue
                counter.domain_pass += 1
                if re.match('(^|[-?+=/_])(big|cyber|hard|huge|mega|small|soft|super|tiny)?(adult|babe|boob|breast|busen|busty|clit|cum|fetish|hooter|lez|lust|naked|nude|porn|porno|pupper|pussy|lesb|gay|lolit|salop|orgasm|mature|sex|smutpump|teen|tit|topp?les|xxx)s?([-.?+=/_]|$)', domain) or re.match('(adultsight|adultsite|adultsonly|adultweb|blowjob|bondage|centerfold|cumshot|cyberlust|cybercore|hardcore|masturbat|obscene|pedophil|pedofil|playmate|pornstar|sexdream|showgirl|softcore|striptease)', domain):
                    continue
                counter.adult_re_pass += 1

                # if not (content_type := record.http_content_type):
                if not (content_type := record.http_headers.get_header('Content-Type')):
                    continue
                record = record.content_stream().read()
                # record = record.reader.read()
                content_type = content_type.lower().strip()
                mark, category = black_list.get(domain, ('normal', ''))

                if group := re.findall('(application|text)/.*(pdf|xml|json|octet-stream)', content_type):
                    _, data_type = group[-1]
                    writer(url, f'extract/{mark}/{data_type}/{domain}/{frag}')
                    data_type = data_type.replace('-', '_')
                    setattr(counter, data_type, getattr(counter, data_type) + 1)
                    continue

                if groups := re.findall('(image|video|audio)/(.*)', content_type):
                    data_type, suffix = groups[-1]
                    writer(url, f'extract/{mark}/{data_type}/{domain}/{frag}')
                    setattr(counter, data_type, getattr(counter, data_type) + 1)
                    continue

                if re.search('text/(html|plain)', content_type):
                    encode = groups[-1] if (groups := re.findall('charset=(.*);', content_type)) else 'utf-8'
                    try:
                        record = str(record, encoding=encode)
                    except LookupError:
                        if 'utf-8' in encode:
                            try:
                                record = str(record, encoding='utf-8')
                            except UnicodeDecodeError:
                                continue
                        else:
                            continue
                    except UnicodeDecodeError:
                        continue
                    counter.encoding_pass += 1
                else:
                    # tqdm.write(colorful.white(f'Ignored Content-Type: {content_type}'))
                    continue

                try:
                    if not (extra := extract(record, url=domain, favor_precision=True, include_comments=True, include_links=True, include_images=True, deduplicate=True)):
                        continue
                except AssertionError:
                    continue
                counter.trafilatura_pass += 1

                try:
                    detect(extra)
                    lang, score = model.predict(extra.replace('\n', ''))
                    lang, score = lang[0][9:], score[0]
                except LangDetectException:
                    continue
                counter.lang_detection_pass += 1

                if not quality_filter(extra, lang, counter, frag):
                    continue

                arrows[f'extract/{mark}/{category}/{lang}/{domain}'].add(extra)
                # tqdm.write((lambda x: colorful.red(x) if mark == 'strict' else colorful.yellow(x) if mark == 'unknown' else colorful.green(x))(f' hit domain:`{domain}` from `{mark}/{category}`'))
                setattr(counter, mark, getattr(counter, mark) + 1)
                counter.language[lang] += 1

                bar.update(1)

        save_items(arrows, frag, counter)
    return counter


non_space_separated = {'zh', 'ja', 'th'}
dir_names = namedtuple('dir_names', ['mark', 'category'])


if __name__ == '__main__':
    colorful = Colorful()
    fasttext.FastText.eprint = lambda _: None
    model = fasttext.load_model('lid.176.bin')
    stopwords = pickle.load(open('stopwords.bin', 'rb'))
    black_list = load_blacklist('blacklist/**/**/domains')

    frags = os.listdir('CC')
    with Pool(min(os.cpu_count(), len(frags))) as pool:
        counters = list(pool.imap_unordered(warc_extract, frags))
        all_counter = sum(counters, start=Counter())
        all_counter.__report__()

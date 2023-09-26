import os
import re
import json
import gzip
import glob
import pickle
import signal
import timeit
import fasttext

from tqdm import tqdm
from tld import get_fld
from utils import Colorful
from trafilatura import extract
from multiprocessing import Pool
from tld.exceptions import TldDomainNotFound
from collections import defaultdict, namedtuple
from warcio.archiveiterator import ArchiveIterator
from langdetect import detect, LangDetectException


class Counter:
    def __enter__(self):
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

        self.response = 0
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
        self.quality_filter_history = defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(colorful.timer(self.start))
        print(colorful.blue(f'{"PDF files":^20}: `{self.pdf:,d}` || rate: `{self.pdf / self.total}`'))
        print(colorful.blue(f'{"XML files":^20}: `{self.xml:,d}` || rate: `{self.xml / self.total}`'))
        print(colorful.blue(f'{"JSON files":^20}: `{self.json:,d}` || rate: `{self.json / self.total}`'))
        print(colorful.blue(f'{"IMAGE files":^20}: `{self.image:,d}` || rate: `{self.image / self.total}`'))
        print(colorful.blue(f'{"VIDEO files":^20}: `{self.video:,d}` || rate: `{self.video / self.total}`'))
        print(colorful.blue(f'{"Octet-Stream files":^20}: `{self.octet_stream:,d}` || rate: `{self.octet_stream / self.total}`'))
        print()
        print(colorful.green(f'{"Total process":^20}: `{self.total:,d}`'))
        print(colorful.blue(f'{"response pass":^20}: `{self.response:,d}` || rate: `{self.response / self.total}`'))
        print(colorful.blue(f'{"domain pass":^20}: `{self.domain_pass:,d}` || rate: `{self.domain_pass / self.total}`'))
        print(colorful.blue(f'{"adult regex pass":^20}: `{self.adult_re_pass:,d}` || rate: `{self.adult_re_pass / self.total}`'))
        print(colorful.blue(f'{"content extract pass":^20}: `{self.trafilatura_pass:,d}` || rate: `{self.trafilatura_pass / self.total}`'))
        print(colorful.blue(f'{"lang detect pass":^20}: `{self.lang_detection_pass:,d}` || rate: `{self.lang_detection_pass / self.total}`'))
        print()
        black_hit = self.unknown + self.useful
        print(colorful.green(f'{"Normal domains":^20}: `{self.normal:,d}` || rate: `{self.normal / self.total}`'))
        print(colorful.green(f'{"Total hit blacklist":^20}: `{black_hit:,d}` || rate `{black_hit / self.total}`'))
        print(colorful.blue(f'{"hit `useful`":^20}: `{self.useful:,d}` || rate `{self.useful / self.total}`'))
        print(colorful.blue(f'{"hit `unknown`":^20}: `{self.unknown:,d}` || rate `{self.unknown / self.total}`'))
        print(colorful.blue(self.language.__str__()))


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


def quality_filter(extra, lang, counter):
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
            counter.quality_filter_history['token_kind_fail'].append({lang: extra})
            return False
        if lang == 'en' and sum(w in stopwords[lang] for w in words) / len(words) < 0.1:
            counter.token_rate_fail += 1
            counter.quality_filter_history['token_rate_fail'].append({lang: extra})
            return False
        if lang == 'en' and sum(all(0x0041 <= ord(s) <= 0x005a or 0x0061 <= ord(s) <= 0x007a for s in w) for w in words) / len(words) < 0.8:
            counter.unicode_rate_fail += 1
            counter.quality_filter_history['unicode_rate_fail'].append({lang: extra})
            return False
    else:
        if not 50 <= len(extra) <= 100000:
            counter.doc_len_fail += 1
            return False
        if len(set(extra)) / len(extra) < 0.2:
            counter.token_kind_fail += 1
            counter.quality_filter_history['token_kind_fail'].append({lang: extra})
            return False
        if lang in 'zh_ja' and sum(s in stopwords[lang] for s in extra) / len(extra) < 0.05:
            counter.token_rate_fail += 1
            counter.quality_filter_history['token_rate_fail'].append({lang: extra})
            return False
        if lang == 'zh' and sum(0x4e00 <= ord(s) <= 0x9fa5 for s in extra) / len(extra) < 0.8:
            counter.unicode_rate_fail += 1
            counter.quality_filter_history['unicode_rate_fail'].append({lang: extra})
            return False
        if lang == 'ja' and sum(0x4e00 <= ord(s) <= 0x9fa5 or 0x3040 <= ord(s) <= 0x309f or 0x30a0 <= ord(s) <= 0x30ff for s in extra) / len(extra) < 0.8:
            counter.unicode_rate_fail += 1
            counter.quality_filter_history['unicode_rate_fail'].append({lang: extra})
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
            counter.quality_filter_history['paragraph_intersection_fail'].append({lang: extra})
            return False
    return True


def writer(url, suffix, record, path):
    name = str(hash(url)) + suffix

    os.makedirs(path, exist_ok=True)
    with open(path + f'/{name}', 'wb') as f:
        f.write(record)
    with open(path + f'/url_index.jsonl', 'a+') as f:
        f.write(json.dumps({name: url}))


def save_items(arrows, frag, counter):
    for path, records in arrows.items():
        os.makedirs(path, exist_ok=True)
        with open(path + f'/{frag}.jsonl', 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps({'data': record}, ensure_ascii=False))

    os.makedirs(path := f'extract/Counter', exist_ok=True)
    with open(path + f'/{frag}', 'wb') as f:
        pickle.dump(counter, f)


def warc_extract(frag):
    arrows = defaultdict(set)
    try:
        with Counter() as counter:
            for file in tqdm(glob.glob(f'CC/{frag}/*.warc.gz'), desc=colorful.blue(frag), colour='BLUE', position=0):
                with gzip.open(file, 'r') as f:
                    for record in tqdm(ArchiveIterator(f), desc=colorful.green(file), position=1):
                        counter.total += 1

                        if record.rec_type != 'response':
                            continue
                        counter.response += 1

                        url = record.rec_headers.get_header('WARC-Target-URI')
                        try:
                            domain = get_fld(url, fix_protocol=True)
                        except TldDomainNotFound:
                            continue
                        if domain in black_list and black_list[domain].mark == 'strict':
                            continue
                        counter.domain_pass += 1
                        if re.match('(^|[-?+=/_])(big|cyber|hard|huge|mega|small|soft|super|tiny)?(adult|babe|boob|breast|busen|busty|clit|cum|fetish|hooter|lez|lust|naked|nude|porn|porno|pupper|pussy|lesb|gay|lolit|salop|orgasm|mature|sex|smutpump|teen|tit|topp?les|xxx)s?([-.?+=/_]|$)', domain) or re.match('(adultsight|adultsite|adultsonly|adultweb|blowjob|bondage|centerfold|cumshot|cyberlust|cybercore|hardcore|masturbat|obscene|pedophil|pedofil|playmate|pornstar|sexdream|showgirl|softcore|striptease)', domain):
                            continue
                        counter.adult_re_pass += 1

                        if not (content_type := record.http_headers.get_header('Content-Type')):
                            continue
                        record = record.content_stream().read()
                        content_type = content_type.lower().strip()
                        mark, category = black_list.get(domain, ('normal', ''))

                        if group := re.findall('(application|text)/.*(pdf|xml|json|octet-stream)', content_type):
                            _, data_type = group[-1]
                            suffix = f'.{data_type}' if data_type != 'octet-stream' else ''
                            writer(url, suffix, record, f'extract/{mark}/{data_type}/{domain}/{frag}')
                            data_type = data_type.replace('-', '_')
                            setattr(counter, data_type, getattr(counter, data_type) + 1)
                            continue

                        if groups := re.findall('(image|video|audio)/(.*)', content_type):
                            data_type, suffix = groups[-1]
                            writer(url, f'.{suffix}', record, f'extract/{mark}/{data_type}/{domain}/{frag}')
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
                            # if lang[:2] != google_lang[:2]:
                            #     tqdm.write(colorful.yellow(f'differ lang fasttext: `{lang}` google: `{google_lang}`'))
                            #     tqdm.write(colorful.yellow(''.join(i for i in extra if i.isprintable())[:100]))
                        except LangDetectException:
                            continue
                        counter.lang_detection_pass += 1

                        if not quality_filter(extra, lang, counter):
                            continue

                        arrows[f'extract/{mark}/{category}/{lang}/{domain}'].add(extra)
                        # tqdm.write((lambda x: colorful.red(x) if mark == 'strict' else colorful.yellow(x) if mark == 'unknown' else colorful.green(x))(f'hit domain:`{domain}` from `{mark}/{category}`'))
                        setattr(counter, mark, getattr(counter, mark) + 1)
                        counter.language[lang] += 1

            save_items(arrows, frag, counter)
    except Exception as e:
        print(frag, e)
        os.killpg(os.getpgid(os.getgid()), signal.SIGKILL)


non_space_separated = {'zh', 'ja', 'th'}
dir_names = namedtuple('dir_names', ['mark', 'category'])


if __name__ == '__main__':
    colorful = Colorful()
    fasttext.FastText.eprint = lambda _: None
    model = fasttext.load_model('lid.176.bin')
    stopwords = pickle.load(open('stopwords.bin', 'rb'))
    black_list = load_blacklist('blacklist/**/**/domains')

    frags = os.listdir('CC')
    with Pool(max(os.cpu_count(), len(frags))) as pool:
        pool.map(warc_extract, frags)

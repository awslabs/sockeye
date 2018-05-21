# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, NamedTuple, Tuple


# Archive types
ARCHIVE_NONE = "none"
ARCHIVE_TAR = "tar"
ARCHIVE_ZIP = "zip"

# Formats for known files
# Note: we currently assume that all data files will be UTF-8 encoded.  If this
#       changes, review the third party tools to make sure everything is
#       converted to UTF-8 immediately after extraction, prior to creating raw
#       train/dev/test files.
TEXT_UTF8_RAW = "utf8_raw"
TEXT_UTF8_RAW_SGML = "utf8_raw_sgml"
TEXT_UTF8_RAW_BITEXT = "utf8_raw_bitext"  # Triple-pipe delimited: source ||| target
TEXT_UTF8_RAW_BITEXT_REVERSE = "utf8_raw_bitext_reverse"  # Same as above, but used
                                                          # for reverse direction
# All TEXT_* types above require tokenization and should appear in this list
TEXT_REQUIRES_TOKENIZATION = [TEXT_UTF8_RAW, TEXT_UTF8_RAW_SGML, TEXT_UTF8_RAW_BITEXT, TEXT_UTF8_RAW_BITEXT_REVERSE]
TEXT_UTF8_TOKENIZED = "utf8_tokenized"


RawFile = NamedTuple("RawFile", [("description", str),
                                 ("url", str),
                                 ("md5", str),
                                 ("archive_type", str)])
"""
Known raw file that provides input data for a sequence-to-sequence task.

:param description: Short description of data contained in raw file.
:param url: Download url.
:param md5: Reference MD5 sum.
:param archive_type: Type of archive, one of ARCHIVE_*.
"""


# Known raw files that provide data for sequence-to-sequence tasks.  Individual
# files may be referenced in multiple tasks.
RAW_FILES = {
    # WMT training data
    "europarl_v7": RawFile("Europarl v7",
                           "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                           "c52404583294a1b609e56d45b2ed06f5",
                           ARCHIVE_TAR),
    "europarl_v8": RawFile("Europarl v8",
                           "http://data.statmt.org/wmt17/translation-task/training-parallel-ep-v8.tgz",
                           "07b77f254d189a5bfb7b43b7fc489716",
                           ARCHIVE_TAR),
    "common_crawl_wmt13": RawFile("Common Crawl corpus (WMT13 release)",
                                  "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                                  "7e0acbe86b0d7816300e14650f5b2bd4",
                                  ARCHIVE_TAR),
    "un_wmt13": RawFile("UN corpus (WMT13 release)",
                        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
                        "bb25a213ba9140023e4cc82c778bef53",
                        ARCHIVE_TAR),
    "news_commentary_v9": RawFile("News Commentary v9",
                                  "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
                                  "92e42b68f9d3c2ae9722e6d1c2623e21",
                                  ARCHIVE_TAR),
    "news_commentary_v12": RawFile("News Commentary v12",
                                   "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
                                   "fc6b83b809347e64f511d291e4bc8731",
                                   ARCHIVE_TAR),
    "giga_fren_wmt10": RawFile("10^9 French-English corpus",
                               "http://www.statmt.org/wmt10/training-giga-fren.tar",
                               "0b12e20027d5b5f0dfcca290c72c8953",
                               ARCHIVE_TAR),
    "wiki_headlines_wmt15": RawFile("Wiki Headlines (WMT15 release)",
                                    "http://www.statmt.org/wmt15/wiki-titles.tgz",
                                    "f74eef43032766d55884a5073ed8ce27",
                                    ARCHIVE_TAR),
    "rapid_eu_2016": RawFile("Rapid corpus of EU press releases (2016)",
                             "http://data.statmt.org/wmt17/translation-task/rapid2016.tgz",
                             "17a3a1846433ad26acb95da02f93af93",
                             ARCHIVE_TAR),
    "leta_v1": RawFile("LETA translated news v1",
                       "http://data.statmt.org/wmt17/translation-task/leta.v1.tgz",
                       "3f367e86924f910cb1e969de57caf63c",
                       ARCHIVE_TAR),
    "dcep_lv_en_v1": RawFile("Digital Corpus of European Parliament v1",
                             "http://data.statmt.org/wmt17/translation-task/dcep.lv-en.v1.tgz",
                             "0f949102e8501dfb3c99d3e3f545b4f9",
                             ARCHIVE_TAR),
    "books_lv_en_v1": RawFile("Online Books v1",
                              "http://data.statmt.org/wmt17/translation-task/books.lv-en.v1.tgz",
                              "7073092421b1259158446870990a9ca5",
                              ARCHIVE_TAR),
    "setimes2_en_tr": RawFile("SETIMES2 English-Turkish",
                              "http://opus.nlpl.eu/download.php?f=SETIMES2/en-tr.txt.zip",
                              "544cec8a631f7820afab6a05451c13a7",
                              ARCHIVE_ZIP),
    # WMT dev and test sets
    "wmt14_dev": RawFile("WMT17 development sets",
                         "http://www.statmt.org/wmt14/dev.tgz",
                         "88ba3fc60b2278d59277122e1c7dd6e7",
                         ARCHIVE_TAR),
    "wmt17_dev": RawFile("WMT17 development sets",
                         "http://data.statmt.org/wmt17/translation-task/dev.tgz",
                         "9b1aa63c1cf49dccdd20b962fe313989",
                         ARCHIVE_TAR),
    "wmt14_test": RawFile("WMT14 test sets",
                          "http://www.statmt.org/wmt14/test-filtered.tgz",
                          "84c597844c1542e29c2aff23aaee4310",
                          ARCHIVE_TAR),
    "wmt17_test": RawFile("WMT17 test sets",
                          "http://data.statmt.org/wmt17/translation-task/test.tgz",
                          "86a1724c276004aa25455ae2a04cef26",
                          ARCHIVE_TAR),
    # Stanford NLP pre-processed data
    "stanford_wmt14_train_en": RawFile("Stanford pre-processed WMT14 English training data",
                                       "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en",
                                       "7ac0d46a8f6db6dfce476c2a8e54121b",
                                       ARCHIVE_NONE),
    "stanford_wmt14_train_de": RawFile("Stanford pre-processed WMT14 German training data",
                                       "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de",
                                       "5873aae4fe517aad42bb29d607b5d2a0",
                                       ARCHIVE_NONE),
    "stanford_wmt14_test2013_en": RawFile("Stanford pre-processed WMT14 English news test 2013",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en",
                                          "f3ce7816bb0acbd2de0364795e9688b1",
                                          ARCHIVE_NONE),
    "stanford_wmt14_test2013_de": RawFile("Stanford pre-processed WMT14 German news test 2013",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de",
                                          "5d48c9300649bfad1300e53ad1334aec",
                                          ARCHIVE_NONE),
    "stanford_wmt14_test2014_en": RawFile("Stanford pre-processed WMT14 English news test 2014",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en",
                                          "4e4663b8de25d19c5fc1c4dab8d61703",
                                          ARCHIVE_NONE),
    "stanford_wmt14_test2014_de": RawFile("Stanford pre-processed WMT14 German news test 2014",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de",
                                          "06e8840abe90cbfbd45cf2729807605d",
                                          ARCHIVE_NONE),
    "stanford_wmt14_test2015_en": RawFile("Stanford pre-processed WMT14 English news test 2015",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en",
                                          "081a724a6a1942eb900d75852f9f5974",
                                          ARCHIVE_NONE),
    "stanford_wmt14_test2015_de": RawFile("Stanford pre-processed WMT14 German news test 2015",
                                          "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de",
                                          "40b6f52962fa630091d8e6a143423385",
                                          ARCHIVE_NONE),
}


Task = NamedTuple("Task", [("description", str),
                           ("url", str),
                           ("src_lang", str),
                           ("trg_lang", str),
                           ("bpe_op", int),
                           ("train", List[Tuple[str, str, str]]),
                           ("dev", List[Tuple[str, str, str]]),
                           ("test", List[Tuple[str, str, str]])])
"""
Sequence-to-sequence task that uses data from known raw files.  Train, dev, and
test files are specified in triples of (source, target, text_type).  The format
for source and target is "raw_file_name/path/to/data/file" and text_type is one
of TEXT_*.  Multiple train and dev sets are concatenated while multiple test
sets are evaluated individually.

:param description: Short description of task.
:param url: URL of task information page.
:param src_lang: Source language code (used for tokenization only).
:param trg_lang: Target language code (used for tokenization only).
:param bpe_op: Number of byte-pair encoding operations for sub-word vocabulary.
:param train: List of training file sets.
:param dev: List of dev/validation file sets.
:param test: List of test/evaluation file sets.
"""


# Known sequence-to-sequence tasks that specify train, dev, and test sets.
TASKS = {
    # WMT14 common benchmarks
    "wmt14_de_en": Task(description="WMT14 German-English news",
                        url="http://www.statmt.org/wmt14/translation-task.html",
                        src_lang="de",
                        trg_lang="en",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.de-en.de",
                             "europarl_v7/training/europarl-v7.de-en.en",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.de-en.de",
                             "common_crawl_wmt13/commoncrawl.de-en.en",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v9/training/news-commentary-v9.de-en.de",
                             "news_commentary_v9/training/news-commentary-v9.de-en.en",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt14_dev/dev/newstest2013-src.de.sgm",
                             "wmt14_dev/dev/newstest2013-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt14_test/test/newstest2014-deen-src.de.sgm",
                             "wmt14_test/test/newstest2014-deen-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt14_fr_en": Task(description="WMT14 French-English news",
                        url="http://www.statmt.org/wmt14/translation-task.html",
                        src_lang="fr",
                        trg_lang="en",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.fr-en.fr",
                             "europarl_v7/training/europarl-v7.fr-en.en",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.fr-en.fr",
                             "common_crawl_wmt13/commoncrawl.fr-en.en",
                             TEXT_UTF8_RAW),
                            ("un_wmt13/un/undoc.2000.fr-en.fr",
                             "un_wmt13/un/undoc.2000.fr-en.en",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v9/training/news-commentary-v9.fr-en.fr",
                             "news_commentary_v9/training/news-commentary-v9.fr-en.en",
                             TEXT_UTF8_RAW),
                            ("giga_fren_wmt10/giga-fren.release2.fixed.fr.gz",
                             "giga_fren_wmt10/giga-fren.release2.fixed.en.gz",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt14_dev/dev/newstest2013-src.fr.sgm",
                             "wmt14_dev/dev/newstest2013-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt14_test/test/newstest2014-fren-src.fr.sgm",
                             "wmt14_test/test/newstest2014-fren-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt14_en_de": Task(description="WMT14 English-German news",
                        url="http://www.statmt.org/wmt14/translation-task.html",
                        src_lang="en",
                        trg_lang="de",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.de-en.en",
                             "europarl_v7/training/europarl-v7.de-en.de",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.de-en.en",
                             "common_crawl_wmt13/commoncrawl.de-en.de",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v9/training/news-commentary-v9.de-en.en",
                             "news_commentary_v9/training/news-commentary-v9.de-en.de",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt14_dev/dev/newstest2013-src.en.sgm",
                             "wmt14_dev/dev/newstest2013-ref.de.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt14_test/test/newstest2014-deen-src.en.sgm",
                             "wmt14_test/test/newstest2014-deen-ref.de.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt14_en_fr": Task(description="WMT14 English-French news",
                        url="http://www.statmt.org/wmt14/translation-task.html",
                        src_lang="en",
                        trg_lang="fr",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.fr-en.en",
                             "europarl_v7/training/europarl-v7.fr-en.fr",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.fr-en.en",
                             "common_crawl_wmt13/commoncrawl.fr-en.fr",
                             TEXT_UTF8_RAW),
                            ("un_wmt13/un/undoc.2000.fr-en.en",
                             "un_wmt13/un/undoc.2000.fr-en.fr",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v9/training/news-commentary-v9.fr-en.en",
                             "news_commentary_v9/training/news-commentary-v9.fr-en.fr",
                             TEXT_UTF8_RAW),
                            ("giga_fren_wmt10/giga-fren.release2.fixed.en.gz",
                             "giga_fren_wmt10/giga-fren.release2.fixed.fr.gz",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt14_dev/dev/newstest2013-src.en.sgm",
                             "wmt14_dev/dev/newstest2013-ref.fr.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt14_test/test/newstest2014-fren-src.en.sgm",
                             "wmt14_test/test/newstest2014-fren-ref.fr.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    # WMT17 tasks using 100% publicly available data
    "wmt17_de_en": Task(description="WMT17 German-English news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="de",
                        trg_lang="en",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.de-en.de",
                             "europarl_v7/training/europarl-v7.de-en.en",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.de-en.de",
                             "common_crawl_wmt13/commoncrawl.de-en.en",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v12/training/news-commentary-v12.de-en.de",
                             "news_commentary_v12/training/news-commentary-v12.de-en.en",
                             TEXT_UTF8_RAW),
                            ("rapid_eu_2016/rapid2016.de-en.de",
                             "rapid_eu_2016/rapid2016.de-en.en",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-deen-src.de.sgm",
                             "wmt17_dev/dev/newstest2016-deen-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-deen-src.de.sgm",
                             "wmt17_test/test/newstest2017-deen-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_fi_en": Task(description="WMT17 Finnish-English news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="fi",
                        trg_lang="en",
                        bpe_op=32000,
                        train=[
                            ("europarl_v8/training/europarl-v8.fi-en.fi",
                             "europarl_v8/training/europarl-v8.fi-en.en",
                             TEXT_UTF8_RAW),
                            ("wiki_headlines_wmt15/wiki/fi-en/titles.fi-en",
                             "wiki_headlines_wmt15/wiki/fi-en/titles.fi-en",
                             TEXT_UTF8_RAW_BITEXT),
                            ("rapid_eu_2016/rapid2016.en-fi.fi",
                             "rapid_eu_2016/rapid2016.en-fi.en",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-fien-src.fi.sgm",
                             "wmt17_dev/dev/newstest2016-fien-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-fien-src.fi.sgm",
                             "wmt17_test/test/newstest2017-fien-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_lv_en": Task(description="WMT17 Latvian-English news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="lv",
                        trg_lang="en",
                        bpe_op=32000,
                        train=[
                            ("europarl_v8/training/europarl-v8.lv-en.lv",
                             "europarl_v8/training/europarl-v8.lv-en.en",
                             TEXT_UTF8_RAW),
                            ("rapid_eu_2016/rapid2016.en-lv.lv",
                             "rapid_eu_2016/rapid2016.en-lv.en",
                             TEXT_UTF8_RAW),
                            ("leta_v1/LETA-lv-en/leta.lv",
                             "leta_v1/LETA-lv-en/leta.en",
                             TEXT_UTF8_RAW),
                            ("dcep_lv_en_v1/dcep.en-lv/dcep.lv",
                             "dcep_lv_en_v1/dcep.en-lv/dcep.en",
                             TEXT_UTF8_RAW),
                            ("books_lv_en_v1/farewell/farewell.lv",
                             "books_lv_en_v1/farewell/farewell.en",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newsdev2017-lven-src.lv.sgm",
                             "wmt17_dev/dev/newsdev2017-lven-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-lven-src.lv.sgm",
                             "wmt17_test/test/newstest2017-lven-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_tr_en": Task(description="WMT17 Turkish-English news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="tr",
                        trg_lang="en",
                        bpe_op=16000,
                        train=[
                            ("setimes2_en_tr/SETIMES2.en-tr.tr",
                             "setimes2_en_tr/SETIMES2.en-tr.en",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-tren-src.tr.sgm",
                             "wmt17_dev/dev/newstest2016-tren-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-tren-src.tr.sgm",
                             "wmt17_test/test/newstest2017-tren-ref.en.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_en_de": Task(description="WMT17 English-German news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="en",
                        trg_lang="de",
                        bpe_op=32000,
                        train=[
                            ("europarl_v7/training/europarl-v7.de-en.en",
                             "europarl_v7/training/europarl-v7.de-en.de",
                             TEXT_UTF8_RAW),
                            ("common_crawl_wmt13/commoncrawl.de-en.en",
                             "common_crawl_wmt13/commoncrawl.de-en.de",
                             TEXT_UTF8_RAW),
                            ("news_commentary_v12/training/news-commentary-v12.de-en.en",
                             "news_commentary_v12/training/news-commentary-v12.de-en.de",
                             TEXT_UTF8_RAW),
                            ("rapid_eu_2016/rapid2016.de-en.en",
                             "rapid_eu_2016/rapid2016.de-en.de",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-ende-src.en.sgm",
                             "wmt17_dev/dev/newstest2016-ende-ref.de.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-ende-src.en.sgm",
                             "wmt17_test/test/newstest2017-ende-ref.de.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_en_fi": Task(description="WMT17 English-Finnish news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="en",
                        trg_lang="fi",
                        bpe_op=32000,
                        train=[
                            ("europarl_v8/training/europarl-v8.fi-en.en",
                             "europarl_v8/training/europarl-v8.fi-en.fi",
                             TEXT_UTF8_RAW),
                            ("wiki_headlines_wmt15/wiki/fi-en/titles.fi-en",
                             "wiki_headlines_wmt15/wiki/fi-en/titles.fi-en",
                             TEXT_UTF8_RAW_BITEXT_REVERSE),
                            ("rapid_eu_2016/rapid2016.en-fi.en",
                             "rapid_eu_2016/rapid2016.en-fi.fi",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-enfi-src.en.sgm",
                             "wmt17_dev/dev/newstest2016-enfi-ref.fi.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-enfi-src.en.sgm",
                             "wmt17_test/test/newstest2017-enfi-ref.fi.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_en_lv": Task(description="WMT17 English-Latvian news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="en",
                        trg_lang="lv",
                        bpe_op=32000,
                        train=[
                            ("europarl_v8/training/europarl-v8.lv-en.en",
                             "europarl_v8/training/europarl-v8.lv-en.lv",
                             TEXT_UTF8_RAW),
                            ("rapid_eu_2016/rapid2016.en-lv.en",
                             "rapid_eu_2016/rapid2016.en-lv.lv",
                             TEXT_UTF8_RAW),
                            ("leta_v1/LETA-lv-en/leta.en",
                             "leta_v1/LETA-lv-en/leta.lv",
                             TEXT_UTF8_RAW),
                            ("dcep_lv_en_v1/dcep.en-lv/dcep.en",
                             "dcep_lv_en_v1/dcep.en-lv/dcep.lv",
                             TEXT_UTF8_RAW),
                            ("books_lv_en_v1/farewell/farewell.en",
                             "books_lv_en_v1/farewell/farewell.lv",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newsdev2017-enlv-src.en.sgm",
                             "wmt17_dev/dev/newsdev2017-enlv-ref.lv.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-enlv-src.en.sgm",
                             "wmt17_test/test/newstest2017-enlv-ref.lv.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    "wmt17_en_tr": Task(description="WMT17 English-Turkish news",
                        url="http://www.statmt.org/wmt17/translation-task.html",
                        src_lang="en",
                        trg_lang="tr",
                        bpe_op=16000,
                        train=[
                            ("setimes2_en_tr/SETIMES2.en-tr.en",
                             "setimes2_en_tr/SETIMES2.en-tr.tr",
                             TEXT_UTF8_RAW),
                        ],
                        dev=[
                            ("wmt17_dev/dev/newstest2016-entr-src.en.sgm",
                             "wmt17_dev/dev/newstest2016-entr-ref.tr.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ],
                        test=[
                            ("wmt17_test/test/newstest2017-entr-src.en.sgm",
                             "wmt17_test/test/newstest2017-entr-ref.tr.sgm",
                             TEXT_UTF8_RAW_SGML),
                        ]),
    # WNMT18 shared task
    "wnmt18_en_de": Task(description="WNMT18 English-German (WMT14 news pre-processed)",
                         url="https://sites.google.com/site/wnmt18/shared-task",
                         src_lang="en",
                         trg_lang="de",
                         bpe_op=32000,
                         train=[
                             ("stanford_wmt14_train_en/train.en",
                              "stanford_wmt14_train_de/train.de",
                              TEXT_UTF8_TOKENIZED),
                         ],
                         dev=[
                             ("stanford_wmt14_test2013_en/newstest2013.en",
                              "stanford_wmt14_test2013_de/newstest2013.de",
                              TEXT_UTF8_TOKENIZED),
                         ],
                         test=[
                             ("stanford_wmt14_test2014_en/newstest2014.en",
                              "stanford_wmt14_test2014_de/newstest2014.de",
                              TEXT_UTF8_TOKENIZED),
                             ("stanford_wmt14_test2015_en/newstest2015.en",
                              "stanford_wmt14_test2015_de/newstest2015.de",
                              TEXT_UTF8_TOKENIZED),
                         ]),
}

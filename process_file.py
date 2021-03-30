import os
import json
from typing import Dict
from collections import defaultdict
from pprint import pprint
from joblib import Parallel, delayed
import re

import pandas as pd
import numpy as np

from diff_processor import DiffProcessor


class DataProcessor:
    """
    We are interested in dropping too long (and probably too short) messages and diffs.
    Current version of this class does the following:
    1) Deletes commit id and issue id
    2) Calculates percentiles
    3) Drops examples out of percentiles
    4) (?) Deletes unchanged lines from diffs
    5) (?) Drops examples with # of tokens in diff > selected number
    and also print some stats.
    """
    def __init__(self,
                 raw_data_dir: str,
                 percentiles_dir: str,
                 processed_data_dir_base: str,
                 stats_dir_base: str,
                 num_workers: int
                 ):

        self.raw_data_dir = raw_data_dir
        self.percentiles_dir = percentiles_dir
        self.processed_data_dir_base = processed_data_dir_base
        self.stats_dir_base = stats_dir_base
        self.num_workers = num_workers

    @staticmethod
    def _calculate_percentiles(filename: str, input_dir: str, output_dir: str):
        """
        This function calculates 1%, 5%, 90%, 95%, 99% percentiles by:
        - number of tokens in diffs
        - number of tokens in messages
        - number of modified files in diff
        for data in `commit_data_dir/filename` and saves resulting json to `output_dir`.
        """
        os.makedirs(output_dir, exist_ok=True)

        # skip already processed files
        if filename.split('.jsonl.gz')[0] + '.json' in os.listdir(output_dir):
            return

        if '.jsonl.gz' not in filename:
            return

        # too huge to fit into memory, leads process to termination, skip for now
        if filename in ['azure_azure-sdk-for-java.jsonl.gz']:
            return

        print(f'Processing {filename}')

        percentiles = {'num_mods': {},
                       'diff_len': {},
                       'message_len': {}}
        try:
            # read json into DataFrame and create some numeric features
            df = pd.read_json(os.path.join(input_dir, filename), compression='gzip', lines=True)
            # remove issue id and commit id
            # issue id: super simple regex to match # 23 (any number)
            # commit id: https://stackoverflow.com/questions/468370/a-regex-to-match-a-sha1
            df['message'] = df['message'].apply(lambda x: re.sub(r'\b([a-f0-9]{7,40})\b', '', re.sub(r'# \d+', '', x)))
            df['diff'] = df['diff'].apply(lambda x: [re.sub(r'\b([a-f0-9]{7,40})\b', '', re.sub(r'# \d+', '', diff)) for diff in x])

            df['num_mods'] = df['diff'].apply(lambda x: len(x))
            df['diff_len'] = df['diff'].apply(lambda x: sum([len(diff.split()) for diff in x]))
            df['message_len'] = df['message'].apply(lambda x: len(x.split()))
            # calculate percentiles
            for key in percentiles:
                for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
                    percentiles[key][q] = df[key].quantile(q)
            # save percentiles
            with open(os.path.join(output_dir,
                                   filename.split('.jsonl.gz')[0] + '.json'), 'w', encoding='utf-8') as f:
                json.dump(percentiles, f, ensure_ascii=False, indent=4)
            print(f'Finished processing {filename}')
            del df
        except ValueError:  # usually not enough memory to load data into json
            print('ValueError with', filename)
        except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
            print('MemoryError with', filename)
        except EOFError:  # usually some errors in file (worth scraping data again)
            print('EOFError with', filename)

    @staticmethod
    def _drop_outliers(filename: str, percentiles: Dict, input_dir: str, output_dir: str):
        """
        This function deletes from `raw_data_dir/filename` examples less or bigger than values from `percentiles`
        and saves result to `output_dit`.
        Specifically:
        - examples with num_mods > 95% percentile are dropped
        - examples with diff_len < 1% percentile are dropped
        - examples with diff_len > 90% percentile are dropped
        - examples with message_len < 5% percentile are dropped
        - examples with message_len > 99% percentile are dropped
        """
        os.makedirs(output_dir, exist_ok=True)

        # skip already processed files
        if filename.split('.jsonl.gz')[0] + '.json' in os.listdir(output_dir):
            return

        if '.jsonl.gz' not in filename:
            return

        # too huge to fit into memory, leads process to termination, skip for now
        if filename in ['azure_azure-sdk-for-java.jsonl.gz']:
            return

        print(f'Processing {filename}')
        try:
            # read json into DataFrame and create some numeric features
            df = pd.read_json(os.path.join(input_dir, filename), compression='gzip', lines=True)

            # remove issue id and commit id
            # issue id: super simple regex to match # 23 (any number)
            # commit id: https://stackoverflow.com/questions/468370/a-regex-to-match-a-sha1
            df['message'] = df['message'].apply(lambda x: re.sub(r'\b([a-f0-9]{7,40})\b', '', re.sub(r'# \d+', '', x)))
            df['diff'] = df['diff'].apply(lambda x: [re.sub(r'\b([a-f0-9]{7,40})\b', '', re.sub(r'# \d+', '', diff)) for diff in x])

            df['num_mods'] = df['diff'].apply(lambda x: len(x))
            df['diff_len'] = df['diff'].apply(lambda x: sum([len(diff.split()) for diff in x]))
            df['message_len'] = df['message'].apply(lambda x: len(x.split()))

            # drop 'outliers'
            df = df.loc[df['num_mods'] <= percentiles['num_mods']['0.95']]

            df = df.loc[df['diff_len'] >= percentiles['diff_len']['0.01']]
            df = df.loc[df['diff_len'] <= percentiles['diff_len']['0.9']]

            df = df.loc[df['message_len'] >= percentiles['message_len']['0.05']]
            df = df.loc[df['message_len'] <= percentiles['message_len']['0.99']]

            # save result to json
            df.to_json(os.path.join(output_dir, filename.split('.jsonl.gz')[0] + '.json'), orient='records')
            print(f'Finished processing {filename}')
            del df
        except ValueError:  # usually not enough memory to load data into json
            print('ValueError with', filename)
        except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
            print('MemoryError with', filename)
        except EOFError:  # usually some errors in file (worth scraping data again)
            print('EOFError with', filename)

    @staticmethod
    def _drop_by_num_tokens(filename: str, diff_max_len: int, input_dir: str, output_dir: str):
        """
        This function deletes from `input_dir/filename` examples with # tokens in diffs > max_len
        and saves result to `output_dir`.
        """
        os.makedirs(output_dir, exist_ok=True)

        # skip already processed files
        if filename in os.listdir(output_dir):
            return

        print(f'Processing {filename}')
        try:
            df = pd.read_json(os.path.join(input_dir, filename), orient='records')

            # remove unchanged lines from diffs
            df = df.loc[df['diff_len'] <= diff_max_len]

            df.to_json(os.path.join(output_dir, filename), orient='records')
            print(f'Finished processing {filename}')
            del df
        except ValueError:  # usually not enough memory to load data into json
            print('ValueError with', filename)
        except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
            print('MemoryError with', filename)
        except EOFError:  # usually some errors in file (worth scraping data again)
            print('EOFError with', filename)

    @staticmethod
    def _calculate_stats(filename: str, input_dir: str, output_dir: str):
        """
        This function calculates some stats for `raw_data_dir/filename` (mean, median, min, max, std, etc.)
        and saves resulting json to `output_dir`.
        """
        os.makedirs(output_dir, exist_ok=True)

        # skip already processed files
        if filename in os.listdir(output_dir):
            return

        print(f'Processing {filename}')

        stats = {'num_mods': {},
                 'diff_len': {},
                 'message_len': {}}
        try:
            df = pd.read_json(os.path.join(input_dir, filename), orient='records')
            df['author'] = df['author'].apply(lambda x: tuple(x))

            # calculate stats for num_mods, diff_len, message_len
            for key in stats:
                stats[key]['mean'] = int(df[key].mean())
                stats[key]['std'] = int(df[key].std())
                stats[key]['median'] = int(df[key].median())
                stats[key]['max'] = int(df[key].max())
                stats[key]['min'] = int(df[key].min())

            # calculate total # of examples and total # of authors
            stats['num_lines'] = len(df)
            stats['num_authors'] = int(df['author'].nunique())

            # calculate stats for # of examples per author
            author_df = df.groupby('author').count()
            stats['ex_per_author'] = {}
            stats['ex_per_author']['mean'] = int(author_df['date'].mean())
            stats['ex_per_author']['median'] = int(author_df['date'].median())
            stats['ex_per_author']['max'] = int(author_df['date'].max())
            stats['ex_per_author']['min'] = int(author_df['date'].min())

            # save result to json
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)

            print(f'Finished processing {filename}')
            del df

        except ValueError:  # usually not enough memory to load data into json
            print('ValueError with', filename)
        except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
            print('MemoryError with', filename)
        except EOFError:  # usually some errors in file (worth scraping data again)
            print('EOFError with', filename)

    @staticmethod
    def _drop_unchanged(filename: str, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # skip already processed files
        if filename in os.listdir(output_dir):
            return

        print(f'Processing {filename}')
        processor = DiffProcessor()
        try:
            df = pd.read_json(os.path.join(input_dir, filename), orient='records')

            # remove unchanged lines from diffs
            df['diff'] = df['diff'].apply(lambda x: [processor.process_diff(diff) for diff in x])
            # re-calculate diff lengths
            df['diff_len'] = df['diff'].apply(lambda x: sum([len(diff.split()) for diff in x]))

            df.to_json(os.path.join(output_dir, filename), orient='records')
            print(f'Finished processing {filename}')
            del df
        except ValueError:  # usually not enough memory to load data into json
            print('ValueError with', filename)
        except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
            print('MemoryError with', filename)
        except EOFError:  # usually some errors in file (worth scraping data again)
            print('EOFError with', filename)

    @staticmethod
    def _aggregate_percentiles(input_dir: str) -> Dict:
        """

        This function creates dict where already calculated percentiles from `percentiles_dir`
        are appended to a single list (to be able to calculate mean value later).
        """
        percentiles = {'num_mods': defaultdict(list),
                       'diff_len': defaultdict(list),
                       'message_len': defaultdict(list)}

        for filename in os.listdir(input_dir):
            try:
                with open(os.path.join(input_dir, filename)) as f:
                    cur_percentiles = json.load(f)
                for key in cur_percentiles:
                    for q in cur_percentiles[key]:
                        percentiles[key][q].append(cur_percentiles[key][q])
            except ValueError:  # usually not enough memory to load data into json
                print('ValueError with', filename)
            except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
                print('MemoryError with', filename)
            except EOFError:  # usually some errors in file (worth scraping data again)
                print('EOFError with', filename)

        return percentiles

    @staticmethod
    def _aggregate_stats(stats_dir: str):
        """
        This function creates dict where already calculated stats from `stats_dir`
        are aggregated (to be able to calculate mean value later).
        """
        stats = {'num_mods': defaultdict(list),
                 'diff_len': defaultdict(list),
                 'message_len': defaultdict(list),
                 'ex_per_author': defaultdict(list),

                 'num_authors': [],

                 'num_lines': 0}

        for filename in os.listdir(stats_dir):
            try:
                with open(os.path.join(stats_dir, filename)) as f:
                    cur_stats = json.load(f)
                for key in cur_stats:
                    if key == 'num_lines':
                        stats[key] += cur_stats[key]
                    elif key == 'num_authors':
                        stats[key].append(cur_stats[key])
                    else:
                        for stat in cur_stats[key]:
                            stats[key][stat].append(cur_stats[key][stat])
            except ValueError:  # usually not enough memory to load data into json
                print('ValueError with', filename)
            except MemoryError:  # usually not enough memory to load json into pandas.DataFrame
                print('MemoryError with', filename)
            except EOFError:  # usually some errors in file (worth scraping data again)
                print('EOFError with', filename)
        return stats

    def calculate_stats(self, data_dir: str, stats_dir: str):
        # calculate stats from each files
        with Parallel(self.num_workers) as pool:
            pool(delayed(DataProcessor._calculate_stats)(filename, data_dir, stats_dir)
                 for filename in os.listdir(data_dir))

        # aggregate stats into final value
        stats = DataProcessor._aggregate_stats(stats_dir)
        for key in stats:
            if key == 'num_lines':
                continue
            elif key == 'num_authors':
                stats[key] = np.mean(stats[key])
            else:
                for stat in stats[key]:
                    stats[key][stat] = np.mean(stats[key][stat])
        return stats

    def calculate_percentiles(self):
        # calculate percentiles from each files
        with Parallel(self.num_workers) as pool:
            pool(delayed(DataProcessor._calculate_percentiles)(filename, self.raw_data_dir, self.percentiles_dir)
                 for filename in os.listdir(self.raw_data_dir))

        # aggregate percentiles into final value
        percentiles = DataProcessor._aggregate_percentiles(self.percentiles_dir)
        for key in percentiles:
            for q in percentiles[key]:
                percentiles[key][q] = np.mean(percentiles[key][q])

        return percentiles

    def drop_outliers(self, percentiles: Dict):
        with Parallel(self.num_workers) as pool:
            pool(delayed(DataProcessor._drop_outliers)(filename, percentiles, self.raw_data_dir,
                                                       self.processed_data_dir_base)
                 for filename in os.listdir(self.raw_data_dir))

    def drop_unchanged(self, input_dir: str, output_dir: str):
        with Parallel(self.num_workers) as pool:
            pool(delayed(DataProcessor._drop_unchanged)(filename,
                                                        input_dir,
                                                        output_dir)
                 for filename in os.listdir(input_dir))

    def drop_by_num_tokens(self, input_dir: str, diff_max_len: int, output_dir: str):
        with Parallel(self.num_workers) as pool:
            pool(delayed(DataProcessor._drop_by_num_tokens)(filename,
                                                            diff_max_len,
                                                            input_dir,
                                                            output_dir)
                 for filename in os.listdir(input_dir))

    def process_data(self):
        # calculate percentiles
        percentiles = self.calculate_percentiles()
        print('===== PERCENTILES =====')
        pprint(percentiles)

        # drop examples out of percentiles
        self.drop_outliers(percentiles)
        stats = self.calculate_stats(self.processed_data_dir_base, self.stats_dir_base)
        print('===== STATS (after removing examples by percentiles) =====')
        pprint(stats)

        # drop examples with num tokens in diff > 2048
        self.drop_by_num_tokens(diff_max_len=2048,
                                input_dir=self.processed_data_dir_base,
                                output_dir=self.processed_data_dir_base + '_short')
        stats = self.calculate_stats(self.processed_data_dir_base + '_short',
                                     self.stats_dir_base + '__short')

        print('===== STATS (after removing diffs with # tokens bigger than 1024) =====')
        pprint(stats)

        # drop unchanged lines from diffs
        self.drop_unchanged(input_dir=self.processed_data_dir_base + '_short',
                            output_dir=self.processed_data_dir_base + '_short_no_unchanged')
        stats = self.calculate_stats(self.processed_data_dir_base + '_short_no_unchanged',
                                     self.stats_dir_base + '_short_no_unchanged')
        print('===== STATS (after removing unchanged lines from diffs) =====')
        pprint(stats)


if __name__ == '__main__':
    raw_data_dir = './extracted_data'
    percentiles_dir = './percentiles'
    processed_data_dir_base = './preprocessed_data'
    stats_dir_base = './stats'

    processor = DataProcessor(raw_data_dir=raw_data_dir,
                              percentiles_dir=percentiles_dir,
                              processed_data_dir_base=processed_data_dir_base,
                              stats_dir_base=stats_dir_base,
                              num_workers=4)
    processor.process_data()




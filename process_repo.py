import os
from typing import List
from dpu_utils.utils import save_jsonl_gz

from joblib import Parallel, delayed, cpu_count

import pydriller

from commit_processor import CommitProcessor


def process_repo(repo_name, repo_url, file_types=['.java']):
    """
    Download author, date, diff and message of all .java-related commits
    and save to .jsonl.gz
    """
    commit_data_dir = 'extracted_data'
    os.makedirs(commit_data_dir, exist_ok=True)

    # skip (for now) as it always leads to crash ^^
    if repo_name == 'aws_aws-sdk-java':
        return

    if repo_name + '.jsonl.gz' in os.listdir(commit_data_dir):
        return

    if repo_name.split('_')[1] in os.listdir('temp'):  # repo is already cloned
        repo = pydriller.RepositoryMining(f'temp/{repo_name.split("_")[1]}', only_no_merge=True,
                                          only_modifications_with_file_types=file_types)
    else:
        try:
            repo = pydriller.RepositoryMining(repo_url, clone_repo_to='temp', only_no_merge=True,
                                              only_modifications_with_file_types=file_types)
        except:
            return
        
    commits_data = []
    
    print(f"Processing {repo_name}")
    
    for i, commit in enumerate(repo.traverse_commits()):
        cur_data = CommitProcessor.process_commit(commit)
        
        if filter_diff(cur_data['diff']) and filter_msg(cur_data['message']):
            commits_data.append(cur_data)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} commits in {repo_name}")
    
    save_jsonl_gz(commits_data, os.path.join(commit_data_dir, f'{repo_name}.jsonl.gz'))    
    print(f"Finished processing {repo_name}")
                          

def filter_diff(diff: List[str], min_len=1) -> bool:
    if len(diff) == 0:
        return False
    if sum(len(x.split()) for x in diff) < min_len:
        return False
    return True
                          

def filter_msg(msg: str, min_len=1) -> bool:
    if len(msg.split()) < min_len:
        return False
    return True


if __name__ == '__main__':
    with open('repos_urls.txt', 'r') as file:
        repo_urls_list = [line.strip() for line in file.readlines()]

    with open('repos_names.txt', 'r') as file:
        repo_names_list = [line.strip() for line in file.readlines()]

    with Parallel(cpu_count()) as pool:
        pool(delayed(process_repo)(repo_name, repo_url) for repo_name, repo_url in zip(repo_names_list, repo_urls_list))

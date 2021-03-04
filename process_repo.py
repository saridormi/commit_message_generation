import pydriller
import os
from dpu_utils.utils import save_jsonl_gz
from typing import List
from commit_processor import CommitProcessor


def process_repo(repo_name, repo_url, num_commits=1000):
    commit_data_dir = 'extracted_data'
    os.makedirs(commit_data_dir, exist_ok=True)

    if repo_name in os.listdir('temp'): # repo is already cloned
        repo = pydriller.RepositoryMining(f'temp/{repo_name}', order='reverse')
    else:
        try:
            repo = pydriller.RepositoryMining(repo_url, clone_repo_to='temp', order='reverse')
        except:
            return {}
        
    commits_data = []
    length = 0
    
    print(f"Processing {repo_name}")
    
    for i, commit in enumerate(repo.traverse_commits()):
        # merge commits are kinda empty
        # https://github.com/ishepard/pydriller/issues/89#issuecomment-590243707
        if commit.merge:  
            continue
            
        cur_data = CommitProcessor.process_commit(commit)
        
        if filter_diff(cur_data['diff']) and filter_msg(cur_data['message']):
            commits_data.append(cur_data)
            length += 1
            if (length + 1) % 100 == 0:
                print(f"Added {length + 1} commits from {repo_name}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} commits in {repo_name}")
        
        if length + 1 >= 1000:
            break
    
    save_jsonl_gz(commits_data, os.path.join(commit_data_dir, f'{repo_name}.jsonl.gz'))    
    print(f"Finished processing {repo_name}")
    return commits_data
                          

def filter_diff(diff: List[str], max_len=512) -> bool:
    if len(diff) == 0:
        return False              
    if sum(len(x.split()) for x in diff) > max_len:
        return False
    return True
                          

def filter_msg(msg: str, max_len=512) -> bool:
    if len(msg.split()) < 8:
        return False
    if len(msg.split()) > max_len:
        return False
    return True
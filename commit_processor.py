import re
import string
import itertools
from datetime import datetime
from typing import Dict, Union, List
from pydriller import ModificationType, Modification, Commit

from diff_processor import DiffProcessor, tokenize_git_diff_output_string


class CommitProcessor:
    @staticmethod
    def preprocess_type(m: Modification) -> str:
        """
        Returns a short summary based on ModificationType to match open dataset data format.
        Strange token <FILE> is needed to distinguish unchanged lines and lines with filename.
        """
        if m.new_path is not None:
            new_path = CommitProcessor.preprocess_diff(m.new_path)
            
        if m.old_path is not None:
            old_path = CommitProcessor.preprocess_diff(m.old_path)
        
        if m.change_type == ModificationType.ADD:
            return f"""new file \n <FILE> {new_path} \n"""

        if m.change_type == ModificationType.RENAME:
            return f"""rename from {old_path} \n rename to {new_path} \n"""

        if m.change_type == ModificationType.DELETE:
            return f"""deleted file \n <FILE> {old_path} \n"""

        if m.change_type == ModificationType.MODIFY:
            return f"""<FILE> {new_path} \n"""

    @staticmethod
    def preprocess_diff(diff: str) -> str:
        """
        Super simple diff processing.
        1) Remove first line with some unnecessary info (e.g.: @@ -0,0 +1,192 @@)
        2) Pad punctuation with spaces
        3) Squeeze multiple spaces to one
        """
        s = re.sub('@@.*@@.*\n', '', diff)
        s = s.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        s = re.sub(' +', ' ', s)
        s = s.strip()
        return s
    
    @staticmethod
    def preprocess_msg(msg: str) -> str:
        """
        Super simple msg processing.
        1) Pad punctuation with spaces
        2) Squeeze multiple spaces to one
        """
        msg_lines = []
        for line in msg.split('\n'):
            line = line.strip('\r')
            if len(line) != 0:
                line = line.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
                line = re.sub(' +', ' ', line)
                line = line.strip()
                msg_lines.append(line)
                
        return ' \\n '.join(msg_lines)

    @staticmethod
    def get_diff_from_modification(m: Modification) -> str:
        """
        1) Generates prefix based on ModificationType
        2) Filters original diff
        3) Returns the concatenation
        """
        if m.change_type in [ModificationType.COPY, ModificationType.UNKNOWN]:
            return None

        prefix = CommitProcessor.preprocess_type(m)
        diff = CommitProcessor.preprocess_diff(m.diff)

        return prefix + diff
    
    @staticmethod
    def process_commit(commit: Commit) -> Dict[str, Union[str, List[str]]]:
        res = {}
        
        res['author'] = (commit.author.name, commit.author.email)
        res['date'] = commit.author_date.strftime("%d.%m.%Y %H:%M:%S")
        res['message'] = CommitProcessor.preprocess_msg(commit.msg)
        res['diff'] = []
        
        for m in commit.modifications:
            diff = CommitProcessor.get_diff_from_modification(m)
            if diff is not None:
                diff = DiffProcessor.process_diff(diff)
                res['diff'].append(diff)
        return res

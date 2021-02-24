import itertools
import os
from typing import Tuple, List


def tokenize_git_diff_output_string(git_diff_output: str) -> List[List[str]]:
    tokens = git_diff_output.split()
    tokens_per_line = [[]]
    for token in tokens:
        if token == '<nl>':
            tokens_per_line.append([])
        else:
            tokens_per_line[-1].append(token)
    if len(tokens_per_line[-1]) == 0:
        tokens_per_line = tokens_per_line[:-1]
    return tokens_per_line


class DiffPreprocessor:
    """Class to process git diff into prev and updated."""
    @staticmethod
    def get_prev_and_updated(git_diff_output: str) -> str:
        """
        Generates prev and updated code from output of git diff command.
        :param git_diff_output: output of git diff command
        :return: [prev, updated]
        """
        tokens_per_line = tokenize_git_diff_output_string(git_diff_output)

        diff_lines = []
        file_before = False
        was_special_keyword_modification = False

        for tokens_in_line in tokens_per_line:
            if tokens_in_line[0] == 'mmm':
                # name of changed file
                # example: mmm a / telecomm / java / android / telecomm / Connection . java
                # example (if new file was created): mmm / dev / null
                if tokens_in_line[1] == 'a':
                    diff_lines.append(tokens_in_line[3:])
                    file_before = True
                was_special_keyword_modification = True

            elif tokens_in_line[0] == 'ppp':
                # name of changed file
                # example: ppp b / telecomm / java / android / telecomm / Connection . java
                # example (if file was deleted):  ppp / dev / null
                if not file_before and tokens_in_line[1] == 'b':
                    diff_lines.append(tokens_in_line[3:])
                was_special_keyword_modification = True

            elif tokens_in_line[:3] == ['new', 'file', 'mode']:
                # line in git diff when new file is created
                # example: new file mode 100644
                diff_lines.append(['new', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:3] == ['deleted', 'file', 'mode']:
                # line in git diff when file is deleted
                # example: deleted file mode 100644
                diff_lines.append(['deleted', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'from']:
                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'to']:
                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['old', 'mode']:
                # line in git diff when file mode was changed
                # example: old mode 100644
                # 644=rw-r--r--
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['new', 'mode']:
                # line in git diff when file mode was changed
                # example: new mode 100755
                # 755=rwxr-xr-x
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[0] == '-':
                # lines that were removed
                # example: - version = ' 2 . 0 . 2 '
                diff_lines.append(tokens_in_line)

            elif tokens_in_line[0] == '+':
                # lines that were added
                # example: + version = ' 2 . 0 . 3 '
                diff_lines.append(tokens_in_line)

            elif tokens_in_line[0] == 'index' or tokens_in_line[:2] == ['similarity', 'index']:
                # some special info that we are not interested in
                # example 1: index 0000000 . . 3f26e45
                # example 2: similarity index 100 %
                continue

            else:
                # all other cases
                # case 1: line that was not changed (drop them)
                # case 2: Binary files a / dependencies / windows / sumatra / SumatraPDF . exe and / dev / null differ
                if tokens_in_line[:2] == ["Binary", "files"]:
                    diff_lines.append(tokens_in_line)

        diff = ' '.join(itertools.chain(*[line + ['\\n'] for line in diff_lines]))

        if not was_special_keyword_modification:
            print(f'No special keyword found for diff: {git_diff_output}')

        return diff + '\n'

    @staticmethod
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[str]:
        diff_res = []
        for git_diff_output in git_diff_outputs:
            diff = DiffPreprocessor.get_prev_and_updated(git_diff_output)
            if diff is not None:
                diff_res.append(diff)
        return diff_res

    @staticmethod
    def create_files(ds_root_path: str):
        for part in ['train', 'val', 'test']:
            cur_path = os.path.join(ds_root_path, part)
            with open(os.path.join(cur_path, 'diff.txt')) as diff_file, \
                 open(os.path.join(cur_path, 'new_diff.txt'), 'w') as new_diff_file:
                diff = DiffPreprocessor.get_prev_and_updated_for_diffs(diff_file.readlines())
                new_diff_file.writelines(diff)


if __name__ == "__main__":
    dataset_path = '../raw_data/CleanedJiang/'
    DiffPreprocessor.create_files(dataset_path)

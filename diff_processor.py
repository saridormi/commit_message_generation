import itertools
import os
from typing import Tuple, List


def tokenize_git_diff_output_string(diff: str) -> List[List[str]]:
    lines = [line.split() for line in diff.split('<nl>')]
    return lines


class DiffProcessor:
    """Class to process git diff."""
    @staticmethod
    def process_diff(diff: str) -> str:
        """
        Removes non-changed lines in diff.
        :param git_diff_output: diff
        :return: processed diff
        """
        tokens_per_line = tokenize_git_diff_output_string(diff)
        diff_lines = []

        for tokens_in_line in tokens_per_line:
            if len(tokens_in_line) == 0:
                #diff_lines.append([]) TODO: might be useful to keep empty lines too?
                continue
                
            elif tokens_in_line[0] == '<FILE>':
                diff_lines.append(tokens_in_line[1:])
                
            elif tokens_in_line[:2] == ['new', 'file']:
                diff_lines.append(tokens_in_line)

            elif tokens_in_line[:2] == ['deleted', 'file']:
                diff_lines.append(tokens_in_line[:2])

            elif tokens_in_line[:2] == ['rename', 'from']:
                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                diff_lines.append(tokens_in_line)

            elif tokens_in_line[:2] == ['rename', 'to']:
                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                diff_lines.append(tokens_in_line)

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

        return diff
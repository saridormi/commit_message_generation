    def get_prev_and_updated(git_diff_output: str) -> str:
        diff_lines = []
        file_before = False
                    diff_lines.append(tokens_in_line[3:])
                    file_before = True
                if not file_before and tokens_in_line[1] == 'b':
                    diff_lines.append(tokens_in_line[3:])
                diff_lines.append(['new', 'file'])
                diff_lines.append(['deleted', 'file'])
                diff_lines.append(tokens_in_line)
                diff_lines.append(tokens_in_line)
                diff_lines.append(tokens_in_line)
                diff_lines.append(tokens_in_line)
                diff_lines.append(tokens_in_line)
                diff_lines.append(tokens_in_line)
                    diff_lines.append(tokens_in_line)
        diff = ' '.join(itertools.chain(*[line + ['\\n'] for line in diff_lines]))
        return diff + '\n'
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[str]:
        diff_res = []
            diff = DiffPreprocessor.get_prev_and_updated(git_diff_output)
            if diff is not None:
                diff_res.append(diff)
        return diff_res
                 open(os.path.join(cur_path, 'new_diff.txt'), 'w') as new_diff_file:
                diff = DiffPreprocessor.get_prev_and_updated_for_diffs(diff_file.readlines())
                new_diff_file.writelines(diff)
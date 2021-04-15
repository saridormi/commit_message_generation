from dataclasses import dataclass
from typing import List, Union, Dict, Optional
from tqdm import tqdm
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorWithHistory:
    """
    Data collator used to
    1) collate author's history with current message
    2) pad and concatenate everything into tensors

    Resulting batch contains keys:
    - diff_input_ids, diff_attention_mask: simply diffs, used for everything
    - msg_input_ids, msg_attention_mask, msg_labels: history + current message, used for training and completion metrics
        - format: history_1 \\n ... history_k \\n cur_msg
    - generation_input_ids, generation_attention_mask: history, used as context/prompt for generation
        - format when history is not empty: <|endoftext|s> history_1 \\n ... history_k \\n
        - format when history is empty: <|endoftext|>
    """

    src_tokenizer: PreTrainedTokenizerBase
    trg_tokenizer: PreTrainedTokenizerBase
    max_len: int

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]],
                                                 List[List[List[int]]],
                                                 torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        diff_inputs = [e['diff_input_ids'] for e in examples]  # 2D - list of lists
        message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists
        history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists (empty/lists of lists)

        all_msg_ids = []  # input for training or metrics: history + cur_msg (right-side padding)
        all_msg_masks = []  # 0 on pad tokens and 1 otherwise (right-side padding)
        all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)

        all_generation_ids = []  # 'prompt' for generation: history (left-side padding)
        all_generation_masks = []  # 0 on pad tokens and 1 otherwise (left-side padding)

        # concatenate history examples with current input ids (checking that resulting length is <= max_len)
        for message_ids, history_ids in zip(message_inputs, history_inputs):
            cur_ids = [message_ids[:self.max_len]]
            cur_labels = [message_ids[:self.max_len]]
            cur_generation_ids = [[self.trg_tokenizer.bos_token_id]]
            cur_len = len(message_ids[:self.max_len])

            for history_input_ids in history_ids[::-1]:
                # insert prev messages from history until we reach max_len
                if cur_len + len(history_input_ids) + len(self.trg_tokenizer(r' \n ').input_ids) > self.max_len:
                    break

                cur_len += len(history_input_ids) + len(self.trg_tokenizer(r' \n ').input_ids)

                cur_ids.insert(0, history_input_ids + self.trg_tokenizer(r' \n ').input_ids)
                cur_generation_ids.insert(1, history_input_ids + self.trg_tokenizer(r' \n ').input_ids)
                cur_labels.insert(0, [-100 for _ in history_input_ids + self.trg_tokenizer(r' \n ').input_ids])

            # flatten everything into one sequence and convert to tensor of torch.int64
            cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_generation_ids = torch.tensor([ex for sublist in cur_generation_ids for ex in sublist],
                                              dtype=torch.int64)
            cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)

            # create ones for attention mask
            cur_mask = torch.ones_like(cur_ids)
            cur_generation_mask = torch.ones_like(cur_generation_ids)

            all_msg_ids.append(cur_ids)
            all_generation_ids.append(cur_generation_ids)
            all_msg_labels.append(cur_labels)
            all_msg_masks.append(cur_mask)
            all_generation_masks.append(cur_generation_mask)

        all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
        all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

        gen_max_len = max(len(tensor) for tensor in all_generation_ids)
        msg_max_len = max(len(tensor) for tensor in all_msg_ids)
        diff_max_len = max(len(tensor) for tensor in all_diff_ids)

        # pad tensors to max length in batch
        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        for i, (diff_id_tensor, diff_mask_tensor,
                msg_id_tensor, msg_mask_tensor, labels_tensor,
                gen_ids_tensor, gen_mask_tensor) in \
                enumerate(zip(all_diff_ids, all_diff_masks,
                              all_msg_ids, all_msg_masks, all_msg_labels,
                              all_generation_ids, all_generation_masks)):
            # pad ids with pad_token_id (which doesn't really matter for GPT-2)
            all_msg_ids[i] = torch.nn.functional.pad(msg_id_tensor, pad=[0, msg_max_len - msg_id_tensor.numel()],
                                                     mode='constant',
                                                     value=self.trg_tokenizer.pad_token_id)
            all_diff_ids[i] = torch.nn.functional.pad(diff_id_tensor, pad=[0, diff_max_len - diff_id_tensor.numel()],
                                                      mode='constant',
                                                      value=self.src_tokenizer.pad_token_id)
            all_generation_ids[i] = torch.nn.functional.pad(gen_ids_tensor,
                                                            pad=[gen_max_len - gen_ids_tensor.numel(), 0],
                                                            mode='constant', value=self.trg_tokenizer.pad_token_id)

            # pad labels with -100
            all_msg_labels[i] = torch.nn.functional.pad(labels_tensor, pad=[0, msg_max_len - labels_tensor.numel()],
                                                        mode='constant', value=-100)

            # pad masks with zeros
            all_msg_masks[i] = torch.nn.functional.pad(msg_mask_tensor, pad=[0, msg_max_len - msg_mask_tensor.numel()],
                                                       mode='constant', value=0)
            all_diff_masks[i] = torch.nn.functional.pad(diff_mask_tensor,
                                                        pad=[0, diff_max_len - diff_mask_tensor.numel()],
                                                        mode='constant', value=0)
            all_generation_masks[i] = torch.nn.functional.pad(gen_mask_tensor,
                                                              pad=[gen_max_len - gen_mask_tensor.numel(), 0],
                                                              mode='constant', value=0)

        all_diff_ids = torch.stack(all_diff_ids)
        all_diff_masks = torch.stack(all_diff_masks)

        all_msg_ids = torch.stack(all_msg_ids)
        all_msg_masks = torch.stack(all_msg_masks)
        all_msg_labels = torch.stack(all_msg_labels)

        all_generation_ids = torch.stack(all_generation_ids)
        all_generation_masks = torch.stack(all_generation_masks)

        return {"diff_input_ids": all_diff_ids,
                "diff_attention_mask": all_diff_masks,
                "msg_input_ids": all_msg_ids,
                "msg_attention_mask": all_msg_masks,
                "msg_labels": all_msg_labels,
                "generation_input_ids": all_generation_ids,
                "generation_attention_mask": all_generation_masks}
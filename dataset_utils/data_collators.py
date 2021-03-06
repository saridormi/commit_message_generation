from dataclasses import dataclass
from typing import List, Union, Dict
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorWithHistory:
    """
    Data collator used to
    1) collate author's history with current message
    2) pad and concatenate everything into tensor
    - diff_input_ids, diff_attention_mask: simply diffs, used for everything
    - msg_input_ids, msg_attention_mask, msg_labels: history + current message, used for training and completion metrics
        - <bos> history_1 \n ... history_k \n cur_msg <eos>
    """

    src_tokenizer: PreTrainedTokenizerBase
    trg_tokenizer: PreTrainedTokenizerBase
    max_len: int
    testing: bool = False

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]],
                                                 List[List[List[int]]],
                                                 torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if not self.testing:
            diff_inputs = [e['diff_input_ids'] for e in examples]  # 2D - list of lists
            message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists
            history_inputs = [e["history_input_ids"] for e in examples]  # 3D - list of lists (empty/lists of lists)

            all_msg_ids = []     # input for training or metrics: history + cur_msg (right-side padding)
            all_msg_masks = []   # 0 on pad tokens and 1 otherwise (right-side padding)
            all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)

            # concatenate history examples with current input ids (checking that resulting length is <= max_len)
            for message_ids, history_ids in zip(message_inputs, history_inputs):
                cur_ids = [[self.trg_tokenizer.bos_token_id],
                           message_ids[:self.max_len - 2],
                           [self.trg_tokenizer.eos_token_id]]
                cur_labels = [[-100], message_ids[:self.max_len - 2], [-100]]
                cur_len = len(message_ids[:self.max_len - 2]) + 2

                for history_input_ids in history_ids[::-1]:
                    # insert prev messages from history until we reach max_len
                    if cur_len + len(history_input_ids) + len(self.trg_tokenizer(r' \n ').input_ids) > self.max_len:
                        break

                    cur_len += len(history_input_ids) + len(self.trg_tokenizer(r' \n ').input_ids)

                    cur_ids.insert(1, history_input_ids + self.trg_tokenizer(r' \n ').input_ids)
                    cur_labels.insert(1, [-100 for _ in history_input_ids + self.trg_tokenizer(r' \n ').input_ids])

                # flatten everything into one sequence and convert to tensor of torch.int64
                cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
                cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)

                # create ones for attention mask
                cur_mask = torch.ones_like(cur_ids)

                all_msg_ids.append(cur_ids)
                all_msg_labels.append(cur_labels)
                all_msg_masks.append(cur_mask)

            all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
            all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

            msg_max_len = max(len(tensor) for tensor in all_msg_ids)
            diff_max_len = max(len(tensor) for tensor in all_diff_ids)

            # pad tensors to max length in batch
            for i, (diff_id_tensor, diff_mask_tensor,
                    msg_id_tensor, msg_mask_tensor, labels_tensor) in \
                    enumerate(zip(all_diff_ids, all_diff_masks,
                                  all_msg_ids, all_msg_masks, all_msg_labels)):
                # pad ids with pad_token_id (which doesn't really matter for GPT-2)
                all_msg_ids[i] = torch.nn.functional.pad(msg_id_tensor, pad=[0, msg_max_len - msg_id_tensor.numel()],
                                                         mode='constant',
                                                         value=self.trg_tokenizer.pad_token_id)
                all_diff_ids[i] = torch.nn.functional.pad(diff_id_tensor, pad=[0, diff_max_len - diff_id_tensor.numel()],
                                                          mode='constant',
                                                          value=self.src_tokenizer.pad_token_id)

                # pad labels with -100
                all_msg_labels[i] = torch.nn.functional.pad(labels_tensor, pad=[0, msg_max_len - labels_tensor.numel()],
                                                            mode='constant', value=-100)

                # pad masks with zeros
                all_msg_masks[i] = torch.nn.functional.pad(msg_mask_tensor, pad=[0, msg_max_len - msg_mask_tensor.numel()],
                                                           mode='constant', value=0)
                all_diff_masks[i] = torch.nn.functional.pad(diff_mask_tensor,
                                                            pad=[0, diff_max_len - diff_mask_tensor.numel()],
                                                            mode='constant', value=0)

            all_diff_ids = torch.stack(all_diff_ids)
            all_diff_masks = torch.stack(all_diff_masks)

            all_msg_ids = torch.stack(all_msg_ids)
            all_msg_masks = torch.stack(all_msg_masks)
            all_msg_labels = torch.stack(all_msg_labels)

            return {"diff_input_ids": all_diff_ids,
                    "diff_attention_mask": all_diff_masks,
                    "msg_input_ids": all_msg_ids,
                    "msg_attention_mask": all_msg_masks,
                    "msg_labels": all_msg_labels}
        else:
            batch_size = len(examples)
            return {"diff_input_ids": torch.randint(10, (batch_size, 500), dtype=torch.int64),
                    "diff_attention_mask": torch.ones(batch_size, 500, dtype=torch.int64),
                    "msg_input_ids": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
                    "msg_attention_mask": torch.ones(batch_size, self.max_len, dtype=torch.int64),
                    "msg_labels": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64)}


@dataclass
class DataCollatorWithoutHistory:
    """
    Data collator used to pad and concatenate everything into tensor

    - diff_input_ids, diff_attention_mask: simply diffs
    - msg_input_ids, msg_attention_mask, msg_labels: current message
        - <bos> cur_msg <eos>
    """

    src_tokenizer: PreTrainedTokenizerBase
    trg_tokenizer: PreTrainedTokenizerBase
    max_len: int
    testing: bool = False

    def __call__(
            self, examples: List[Dict[str, Union[List[List[int]],
                                                 List[List[List[int]]],
                                                 torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if not self.testing:
            diff_inputs = [e['diff_input_ids'] for e in examples]  # 2D - list of lists
            message_inputs = [e["msg_input_ids"] for e in examples]  # 2D - list of lists

            all_msg_ids = []     # input for training or metrics: history + cur_msg (right-side padding)
            all_msg_masks = []   # 0 on pad tokens and 1 otherwise (right-side padding)
            all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)

            for message_ids in message_inputs:
                cur_ids = [[self.trg_tokenizer.bos_token_id],
                           message_ids[:self.max_len - 2],
                           [self.trg_tokenizer.eos_token_id]]
                cur_labels = [[-100], message_ids[:self.max_len - 2], [-100]]

                # flatten everything into one sequence and convert to tensor of torch.int64
                cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
                cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)

                # create ones for attention mask
                cur_mask = torch.ones_like(cur_ids)

                all_msg_ids.append(cur_ids)
                all_msg_labels.append(cur_labels)
                all_msg_masks.append(cur_mask)

            all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in diff_inputs]
            all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

            msg_max_len = max(len(tensor) for tensor in all_msg_ids)
            diff_max_len = max(len(tensor) for tensor in all_diff_ids)

            # pad tensors to max length in batch
            for i, (diff_id_tensor, diff_mask_tensor,
                    msg_id_tensor, msg_mask_tensor, labels_tensor) in \
                    enumerate(zip(all_diff_ids, all_diff_masks,
                                  all_msg_ids, all_msg_masks, all_msg_labels)):
                # pad ids with pad_token_id (which doesn't really matter for GPT-2)
                all_msg_ids[i] = torch.nn.functional.pad(msg_id_tensor, pad=[0, msg_max_len - msg_id_tensor.numel()],
                                                         mode='constant',
                                                         value=self.trg_tokenizer.pad_token_id)
                all_diff_ids[i] = torch.nn.functional.pad(diff_id_tensor, pad=[0, diff_max_len - diff_id_tensor.numel()],
                                                          mode='constant',
                                                          value=self.src_tokenizer.pad_token_id)

                # pad labels with -100
                all_msg_labels[i] = torch.nn.functional.pad(labels_tensor, pad=[0, msg_max_len - labels_tensor.numel()],
                                                            mode='constant', value=-100)

                # pad masks with zeros
                all_msg_masks[i] = torch.nn.functional.pad(msg_mask_tensor, pad=[0, msg_max_len - msg_mask_tensor.numel()],
                                                           mode='constant', value=0)
                all_diff_masks[i] = torch.nn.functional.pad(diff_mask_tensor,
                                                            pad=[0, diff_max_len - diff_mask_tensor.numel()],
                                                            mode='constant', value=0)

            all_diff_ids = torch.stack(all_diff_ids)
            all_diff_masks = torch.stack(all_diff_masks)

            all_msg_ids = torch.stack(all_msg_ids)
            all_msg_masks = torch.stack(all_msg_masks)
            all_msg_labels = torch.stack(all_msg_labels)

            return {"diff_input_ids": all_diff_ids,
                    "diff_attention_mask": all_diff_masks,
                    "msg_input_ids": all_msg_ids,
                    "msg_attention_mask": all_msg_masks,
                    "msg_labels": all_msg_labels}
        else:
            batch_size = len(examples)
            return {"diff_input_ids": torch.randint(10, (batch_size, 500), dtype=torch.int64),
                    "diff_attention_mask": torch.ones(batch_size, 500, dtype=torch.int64),
                    "msg_input_ids": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64),
                    "msg_attention_mask": torch.ones(batch_size, self.max_len, dtype=torch.int64),
                    "msg_labels": torch.randint(10, (batch_size, self.max_len), dtype=torch.int64)}

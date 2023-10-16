"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of `tk_ids` and `attn_masks`, should be `torch.int64`, and the updated type of `lbs` should be `torch.float32`.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---
        max_len = -1
        for tk in tk_ids:
            max_len = max(max_len, len(tk))

        pad_token = 0
        batch_tk_ids = torch.zeros((len(tk_ids),max_len))
        for i, tk_id in enumerate(tk_ids):
            batch_tk_ids[i][:len(tk_id)] = torch.Tensor(tk_id)
            batch_tk_ids[i][len(tk_id):] = pad_token
        tk_ids = batch_tk_ids.type(torch.int64)

        pad_token = 0
        batch_attn_masks = torch.zeros((len(attn_masks),max_len))
        for i, attn_mask in enumerate(attn_masks):
            batch_attn_masks[i][:len(attn_mask)] = torch.Tensor(attn_mask)
            batch_attn_masks[i][len(attn_mask):] = pad_token
        attn_masks = batch_attn_masks.type(torch.int64)

        pad_token = self.label_pad_token_id
        batch_lbs = torch.zeros((len(lbs),max_len))
        for i, lb in enumerate(lbs):
            batch_lbs[i][:len(lb)] = torch.Tensor(lb)
            batch_lbs[i][len(lb):] = pad_token
        lbs = batch_lbs.type(torch.int64)
        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)

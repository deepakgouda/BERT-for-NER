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

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)

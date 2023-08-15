import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTagDataset(Dataset):
    def __init__(self, quest, tags, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = quest.tolist()
        self.labels = tags.tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item_idx):
        text = self.text[item_idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length= self.max_len,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'
            )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        labels = torch.tensor(self.labels[item_idx], dtype=torch.float, device=device, requires_grad=False)
        result = {
            'input_ids': input_ids ,
            'attention_mask': attn_mask,
            'label': labels
            }
        
        return result
    

class QTagDataModule(pl.LightningDataModule):
    def __init__(self, x_tr, y_tr, x_val, y_val, x_test, y_test,tokenizer,batch_size=16,max_token_len=200):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = QTagDataset(quest=self.tr_text, tags=self.tr_label, tokenizer=self.tokenizer, max_len=self.max_token_len)
        self.val_dataset  = QTagDataset(quest=self.val_text, tags=self.val_label, tokenizer=self.tokenizer, max_len=self.max_token_len)
        self.test_dataset  = QTagDataset(quest=self.test_text, tags=self.test_label, tokenizer=self.tokenizer,max_len=self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
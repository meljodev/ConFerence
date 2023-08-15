from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel

import torch
from torch import nn
import pytorch_lightning as pl


class QTagClassifier(pl.LightningModule):

    def __init__(self, n_classes, steps_per_epoch=None, n_epochs=3, lr=2e-5):
        super().__init__()
        self.model = AutoModel.from_pretrained("rasa/LaBSE")
        # self.model = model
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters('n_classes')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output['pooler_output']
        output = self.classifier(pooled_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        loss, outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss" :loss, "predictions":outputs, "labels": labels }

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        loss, outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        loss, outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]
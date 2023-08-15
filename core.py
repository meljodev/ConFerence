import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from config import config
from data_module import QTagDataModule
from classifier_module import QTagClassifier
from process import get_dataset

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

# data = pd.read_json(config.JSON_DATASET_PATH)

# data = data.replace('Customer:', ' ', regex=True)
# data = data.replace('Operator:', ' ', regex=True)
data = get_dataset()

mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(data['gpt_intents'])
class_nums = len(mlb.classes_)

x = data['conversation_content']

x_train, x_test, y_train, y_test = train_test_split(x, yt, test_size=0.1, random_state=config.RANDOM_SEED, shuffle=False)
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=config.RANDOM_SEED, shuffle=False)

checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='QTag-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min')

def classify(pred_prob, thresh):
    y_pred = []

    for tag_label_row in pred_prob:
        temp=[]
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1) 
            else:
                temp.append(0)
        y_pred.append(temp)
    return y_pred


def train(tokenizer):
    QTdata_module = QTagDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer, config.BATCH_SIZE)
    QTdata_module.setup()

    steps_per_epoch = len(x_tr)//config.BATCH_SIZE
    model = QTagClassifier(n_classes=class_nums, steps_per_epoch=steps_per_epoch,
                           n_epochs=config.N_EPOCHS, lr=config.LEARNING_RATE)

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                        max_epochs=config.N_EPOCHS,
                        enable_progress_bar=True)
    
    trainer.fit(model=model, datamodule=QTdata_module)
    trainer.test(model=model, datamodule=QTdata_module)
    
    return model

def prepare_test_set(tokenizer):
    input_ids = []
    attention_masks = []

    for quest in x_test:
        encoded_quest = tokenizer.encode_plus(
                        quest,
                        None,
                        add_special_tokens=True,
                        max_length=config.MAX_LEN,
                        padding = 'max_length',
                        return_token_type_ids= False,
                        return_attention_mask= True,
                        truncation=True,
                        return_tensors = 'pt')
 
        input_ids.append(encoded_quest['input_ids'])
        attention_masks.append(encoded_quest['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y_test)

    pred_data = TensorDataset(input_ids, attention_masks, labels)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=config.TEST_BATCH_SIZE)
    
    return pred_dataloader

def test(model, tokenizer):
    pred_dataloader= prepare_test_set(tokenizer)
    flat_pred_outs = 0
    flat_true_labels = 0
    model = model.to(config.DEVICE)
    model.eval()
    pred_outs, true_labels = [], []

    for batch in pred_dataloader:
        batch = tuple(t.to(config.DEVICE) for t in batch)
        b_input_ids, b_attn_mask, b_labels = batch

        with torch.no_grad():
            loss, pred_out = model(b_input_ids,b_attn_mask)
            pred_out = pred_out.detach().cpu().numpy()
            label_ids = b_labels.to(config.DEVICE).numpy()

        pred_outs.append(pred_out)
        true_labels.append(label_ids)

    flat_pred_outs = np.concatenate(pred_outs, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    threshold  = np.arange(0.4, 0.6)

    scores=[] 
    y_true = flat_true_labels.ravel() 

    for thresh in threshold:
        pred_bin_label = classify(flat_pred_outs,thresh) 
        y_pred = np.array(pred_bin_label).ravel()
        scores.append(metrics.f1_score(y_true, y_pred))

    opt_thresh = threshold[scores.index(max(scores))]
    print(f'Optimal Threshold Value = {opt_thresh}')

    y_pred_labels = classify(flat_pred_outs,opt_thresh)
    y_pred = np.array(y_pred_labels).ravel() 
    print(metrics.classification_report(y_true,y_pred))
    y_pred = mlb.inverse_transform(np.array(y_pred_labels))
    y_act = mlb.inverse_transform(flat_true_labels)

    df = pd.DataFrame({'Body':x_test,'Actual Intents':y_act,'Predicted Intents':y_pred})

    print("test result df : \n", df)
    
    return opt_thresh

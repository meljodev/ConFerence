import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from config import config
from classifier_module import QTagClassifier
from core import mlb

def inference(input_text, tokenizer, checkpoint, threshold):
    model_path = "lightning_logs/version_2/checkpoints/QTag-epoch=02-val_loss=0.81.ckpt"
    intents = predict_intent(input_text, tokenizer, model_path, threshold)
    if not intents[0]:
        print('This conversation can not be associated with any known intent - Please review to see if a new intent is required ')
        return [("other_intent")]
    else:
        print(f'Following intents are associated : \n {intents}')
        print(type(intents))
        return intents
    

def predict_intent(question, tokenizer, model_path, threshold):
    QTmodel = QTagClassifier.load_from_checkpoint(model_path)
    # import torch
    # QTmodel = torch.load
    QTmodel.eval()
    text_enc = tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'      
    )
    loss, outputs = QTmodel(text_enc['input_ids'], text_enc['attention_mask'])
    pred_out = outputs[0].detach().numpy()
    print(pred_out)
    preds = [(pred > threshold) for pred in pred_out ]
    preds = np.asarray(preds)
    new_preds = preds.reshape(1,-1).astype(int)
    print(new_preds)
    pred_tags = mlb.inverse_transform(new_preds)
    print(mlb.inverse_transform(np.array(new_preds)))
    print(type(pred_tags))
    return pred_tags 


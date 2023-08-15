import json, os
import pandas as pd

def get_dataset(): 
    path_to_json_files = 'gpt_result/'
    json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]

    combined_batch_jsons = []
    for json_file_name in json_file_names:
        with open(os.path.join(path_to_json_files, json_file_name), encoding="utf-8") as json_file:
            json_batch = json.load(json_file)
            for row in json_batch:
                combined_batch_jsons.append(row)

    dataset = pd.DataFrame(combined_batch_jsons)
    del dataset['conversation_date']
    del dataset['id']
    dataset.dropna(subset=['gpt_intents'], inplace=True)
    dataset['gpt_intents'] = dataset['gpt_intents'].apply(lambda intents: [intent['intent'] for intent in intents])
    dataset = dataset.replace('Customer:', ' ', regex=True)
    dataset = dataset.replace('Operator:', ' ', regex=True)
    return dataset



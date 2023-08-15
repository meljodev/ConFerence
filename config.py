import torch

class Config:
    # Dataset path
    JSON_DATASET_PATH: str = "result.json"
    
    # Model hyperparameters
    RANDOM_SEED: int = 42
    N_EPOCHS: int = 4
    BATCH_SIZE: int = 4
    MAX_LEN: int = 300
    LEARNING_RATE: int = 2e-05
    TEST_BATCH_SIZE = 64 
    
    # Pretrained model name
    PRETRAINED_MODEL = "rasa/LaBSE"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config=Config()
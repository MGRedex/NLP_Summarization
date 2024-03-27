import torch

config = {
    "SHUFFLE_SEED": 42,
    "DATA": {
        "BATCH_SIZE": 32,
        "TRAIN_SAMPLES": range(32*10000),
        "VALIDATION_SAMPLES": range(32*1000),
        "DATALOADER_NUM_WORKERS": 4,
        "NON_BLOCKING": True,
    },
    "MODEL": {
        "EMB_DIM": 256,
        "SEQ_LEN": 256,
        "DROPOUTS": {
            "POS_EMB": 0.1,
            "ENCODER": 0.1,
            "DECODER": 0.1,
        },
        "ATTN_HEADS": 8,
        "FF_DIM": 512,
        "DTYPE": torch.bfloat16,
        "WARMUP": True,
        "TF32": False,
    },
    "TRAINING": {
        "EPOCH": 10,
        "LOGS_FOLDER": "./logs",
        "CHECKPOINT_BY": "BLEU",
        "CHECKPOINT_PATH":"./train_states/best_state_6dec.pt", 
    },
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

default_config = {
    "SHUFFLE_SEED": 42,
    "DATA": {
        "BATCH_SIZE": 32,
        "TRAIN_SAMPLES": range(32*1000),
        "VALIDATION_SAMPLES": range(32*1000),
        "DATALOADER_NUM_WORKERS": 4,
        "NON_BLOCKING": True,
    },
    "MODEL": {
        "EMB_DIM": 256,
        "SEQ_LEN": 256,
        "DROPOUTS": {
            "POS_EMB": 0.1,
            "ENCODER": 0.1,
            "DECODER": 0.1,
        },
        "ATTN_HEADS": 8,
        "FF_DIM": 512,
        "DTYPE": torch.bfloat16,
        "WARMUP": True,
        "TF32": False,
    },
    "TRAINING": {
        "EPOCH": 10,
        "LOGS_FOLDER": "./logs",
        "CHECKPOINT_BY": "BLEU",
        "CHECKPOINT_PATH":"./train_states/best_state_5dec.pt", 
    },
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from .neural import Transformer

def create_run_logger(
        folder: str,
        model: nn.Module
    ) -> SummaryWriter:
    """Creates SummaryWriter for model training run
    that writes into folder\\model_name\\run_number
    where:

        folder - log folder

        model_name - name of model's class

        run_number - next run number (if there are previous run folders) or 0
    """
    run_folder = f"{folder}/{model.__class__.__name__}"
    try:
        run_n = int(max(os.listdir(run_folder)))
        run_n += 1
    except:
        run_n = 0
    return SummaryWriter(f"{run_folder}/{run_n}")

def test_model(config, train_dataloader):
    DEVICE = config["DEVICE"]
    if config["MODEL"]["TF32"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_tokenizer.add_special_tokens({
        "eos_token": "[EOS]",
        "bos_token": "[BOS]"
    })

    model = Transformer(
        vocab_size = len(bert_tokenizer),
        seq_len = config["MODEL"]["SEQ_LEN"],
        emb_dim = config["MODEL"]["EMB_DIM"],
        n_heads = config["MODEL"]["ATTN_HEADS"],
        feedforward_dim = config["MODEL"]["FF_DIM"],
        dropouts = config["MODEL"]["DROPOUTS"],
        dtype = config["MODEL"]["DTYPE"],
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = Adam(params = model.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps = 1e-9)

    scheduler = ExponentialLR(optimizer, gamma = 0.8)
    # model_metrics = {"BLEU": metrics.BLEUScore(n_gram = 4)}

    # WARMUP
    if config["MODEL"]["WARMUP"]:
        max_doc_len = 191
        max_sum_len = 80

        X = torch.rand(config["DATA"]["BATCH_SIZE"],max_doc_len).type(torch.LongTensor).to(DEVICE)
        y = torch.rand(config["DATA"]["BATCH_SIZE"],max_sum_len).type(torch.LongTensor).to(DEVICE)
        summary_with_eos = torch.rand(config["DATA"]["BATCH_SIZE"],max_sum_len,1).type(torch.LongTensor).to(DEVICE)

        y_pred = model(X, y)
        loss = loss_fn(y_pred.view(-1, y_pred.shape[2]), summary_with_eos.view(-1))
        loss.backward()

        optimizer.zero_grad()

    print(f'model: {model.__class__.__name__}, epochs: {config["TRAINING"]["EPOCH"]}, batch size: {config["DATA"]["BATCH_SIZE"]}, train dataset len: {len(train_dataloader) * config["DATA"]["BATCH_SIZE"]} inst.')
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        for epoch in range(config["TRAINING"]["EPOCH"]):
            print(f"epoch: {epoch}")
            ov_loss = 0
            model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                x, y = batch["document"].to(DEVICE, non_blocking = config["DATA"]["NON_BLOCKING"]), batch["summary"].to(DEVICE, non_blocking = config["DATA"]["NON_BLOCKING"])

                y_pred = model(x, y)

                summary_with_eos = F.pad(y[:, 1:], pad = (0,1), value = 0)
                summary_with_eos[torch.arange(config["DATA"]["BATCH_SIZE"], device = DEVICE), summary_with_eos.argmin(dim = 1)] = bert_tokenizer.eos_token_id

                loss = loss_fn(y_pred.view(-1, y_pred.shape[2]), summary_with_eos.view(-1))
                ov_loss += loss
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            print(f"overall loss: {loss / len(train_dataloader)}")
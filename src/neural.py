from typing import Any
import torch
from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.optim import Adam
from transformers import AutoTokenizer
import math
from tqdm import tqdm
from typing import (
    Optional,
    Dict 
)
import lightning as L
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            seq_len: int,
            dropout: float,
            dtype = None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if dtype:
            pe = pe.type(dtype)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)

# https://github.com/alex-matton/causal-transformer-decoder
class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        self.autoregressive_inf = False
        super().__init__(*args, **kwargs)

    def eval(self, autoregressive = False):
        self.autoregressive_inf = autoregressive
        return self.train(False)

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Tensor = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """

        if self.training or (not self.training and not self.autoregressive_inf):
            return super().forward(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        tgt_last_tok = tgt[:, -1:, :]

        # self attention part
        tgt_last_tok = self.norm1(tgt_last_tok + self.dropout1(
            self.self_attn(
                tgt_last_tok,
                tgt,
                tgt,
                attn_mask=None,  # not needed because we only care about the last token
                key_padding_mask=tgt_key_padding_mask,
                need_weights = False,
            )[0])
        )

        # encoder-decoder attention
        if memory is not None:
            tgt_last_tok = self.norm2(tgt_last_tok + self.dropout2(
                self.multihead_attn(
                    tgt_last_tok,
                    memory,
                    memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    need_weights = False,
                )[0])
            )

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok

class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """
    def __init__(self, *args, **kwargs):
        self.autoregressive_inf = False
        super().__init__(*args, **kwargs)

    def eval(self, autoregressive = False):
        self.autoregressive_inf = autoregressive
        for module in self.modules():
            if isinstance(module, CausalTransformerDecoderLayer):
                module.autoregressive_inf = autoregressive
        return self.train(False)

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Tensor = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training or (not self.training and not self.autoregressive_inf):
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    tgt_mask = tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            return output

        new_token_cache = []

        # Iterate through all decoder layers in stack
        for i, mod in enumerate(self.layers):

            # Generate one new token embedding per batch
            output = mod(output, memory, memory_mask = memory_mask)

            # Write new token embedding in new_token_cache
            new_token_cache.append(output)

            # If it's not first iteration in generation
            # then there is a cached state for next decoder
            # so we provide that state + new token embedding from current decoder
            if cache is not None:
                output = torch.cat([cache[i], output], dim=1)

        # If it's not first iteration in generation
        # and we have cached intermediate states for decoders
        # we add new token embeddings from each decoder to old cache
        # (we did the same but only for one state in the cycle)
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=2)
        # If it's first iteration in generation
        # we create cache based on token embeddings from each decoder
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        # output = last_decoder_cached_state + new token embedding
        # so we could it provide to first decoder in the next iteration
        # but we add last transformed(to token index) embeddings to targets outside the decoder stack
        # so we need only last token
        return output[:, -1:, :], new_cache

class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            emb_dim: int,
            n_heads: int,
            feedforward_dim: int,
            dropouts: Dict[str,float],
            dtype = None,
            activation = nn.LeakyReLU(),
            dec_num = 3,
            enc_num = 3,
    ):
        super().__init__()

        # Layers
        self.embedder = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim,
            padding_idx = 0,
            dtype = dtype,
        )
        self.pos_embedder = PositionalEncoding(
            emb_dim = emb_dim,
            seq_len = seq_len,
            dropout = dropouts["POS_EMB"],
            dtype = dtype,
        )
        encoder = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dim_feedforward = feedforward_dim,
            batch_first = True,
            dropout = dropouts["ENCODER"],
            activation = activation,
            dtype = dtype,
        )
        decoder = CausalTransformerDecoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dim_feedforward = feedforward_dim,
            batch_first = True,
            dropout = dropouts["DECODER"],
            activation = activation,
            dtype = dtype,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer = encoder,
            num_layers = enc_num,
        )
        self.decoder_stack = CausalTransformerDecoder(
            decoder_layer = decoder,
            num_layers = dec_num,
        )
        self.output_linear = nn.Linear(
            in_features = emb_dim,
            out_features = vocab_size,
            dtype = dtype,
        )

        # Attributes
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim
        self.autoregressive_inf = False

        # Buffers
        self.register_buffer(
            "peak_ahead_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool(),
        )
        
    def eval(self, autoregressive = False):
        self.autoregressive_inf = autoregressive
        self.decoder_stack.eval(autoregressive)
        return self.train(False)

    def forward(
            self,
            src: Tensor,
            tgt: Optional[Tensor] = None,
            pad_token: int = 0,
            eos_token: int = 30522,
            bos_token: int = 30523,
    ):    
        # Compute padding masks
        src_pad_mask = (src == pad_token)

        # Create encoded src once
        encoded = self.encoder_stack(
            self.pos_embedder(self.embedder(src)),
            src_key_padding_mask = src_pad_mask
        )

        # If training mode on, whole output is generated based on whole target
        if self.training or (not self.training and not self.autoregressive_inf):
            tgt_pad_mask = (tgt == pad_token)
            tgt_length = len(tgt[-1])
            peak_ahead_mask = self.peak_ahead_mask[:tgt_length, :tgt_length]
            return self.output_linear(
                self.decoder_stack(
                    self.pos_embedder(self.embedder(tgt)),
                    memory = encoded,
                    tgt_mask = peak_ahead_mask,
                    tgt_key_padding_mask = tgt_pad_mask,
                    memory_key_padding_mask = src_pad_mask
                )
            )
        # If eval mode on, output is generated one token at a time based on last predicted token as query and all predicted tokens as keys, values
        else:
            # Create target batch, with [BOS] first token 
            tgt = torch.tensor(encoded.shape[0]*[[bos_token]], device = next(self.parameters()).device)

            # Each target with [EOS] token is output, so it moved to preds 
            preds = []

            # If we have decoder stack, then for first decoder we can provide
            # last and all previous predicted tokens
            # but there are intermediate states between decoders
            # we have to cache, so each decoder gets last and all previous predicted from previous decoder
            cache = None

            # Generate one token at a time for each target in batch
            for _ in range(0,self.seq_len):
                # If all targets moved to preds, generating is ended
                if len(tgt) == 0: break

                # New tokens(batch_size, 1, emb_dim), previous intermediate states + new state(num_decoders, batch_size, state_length, emb_dim)
                decoder_out, cache = self.decoder_stack(
                        self.pos_embedder(self.embedder(tgt)),
                        memory = encoded,
                        memory_key_padding_mask = src_pad_mask,
                        cache = cache
                )

                # Get probability dist.(batch_size, 1, vocab_size)
                distribution = self.output_linear(decoder_out)

                # Add new tokens with highest probabilities to tgt
                tgt = torch.cat([tgt, distribution.argmax(dim = -1)], dim = 1)

                # Get indexes of targets with [EOS] token
                predicted = (tgt[:,-1] == eos_token)

                # If there are targets with [EOS] token
                # Move targets to preds
                # Remove targets from tgt
                # Remove targets intermediate states from cache
                # Remove targets srcs
                if predicted.any():
                    preds.extend(list(tgt[predicted]))
                    tgt = tgt[~predicted]
                    cache = cache[:, ~predicted]
                    encoded = encoded[~predicted]

            # If we have targets with max length and without [EOS], they are not in preds
            if len(tgt) != 0:
                preds.extend(list(tgt))

            return preds 
            # return tgt

class LightningTransformer(L.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            emb_dim: int,
            n_heads: int,
            feedforward_dim: int,
            dropouts: Dict[str,float],
            dtype = None,
            special_tokens: Dict = {
                "pad_token": 0,
                "eos_token": 30522,
                "bos_token": 30523,
            }
    ) -> None:
        super().__init__()

        # Layers
        self.embedder = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim,
            padding_idx = 0,
            dtype = dtype,
        )
        self.pos_embedder = PositionalEncoding(
            emb_dim = emb_dim,
            seq_len = seq_len,
            dropout = dropouts["POS_EMB"],
            dtype = dtype,
        )
        encoder = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dim_feedforward = feedforward_dim,
            batch_first = True,
            dropout = dropouts["ENCODER"],
            dtype = dtype,
        )
        decoder = CausalTransformerDecoderLayer(
            d_model = emb_dim,
            nhead = n_heads,
            dim_feedforward = feedforward_dim,
            batch_first = True,
            dropout = dropouts["DECODER"],
            dtype = dtype,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer = encoder,
            num_layers = 3,
        )
        self.decoder_stack = CausalTransformerDecoder(
            decoder_layer = decoder,
            num_layers = 3,
        )
        self.output_linear = nn.Linear(
            in_features = emb_dim,
            out_features = vocab_size,
            dtype = dtype,
        )

        # Attributes
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.feedforward_dim = feedforward_dim

        # Buffers
        self.register_buffer(
            "peak_ahead_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool(),
        )

        self.special_tokens = special_tokens

    def training_step(self, batch, batch_idx):
        X, y = batch["document"].to(config["DEVICE"], non_blocking = True), batch["summary"].to(config["DEVICE"], non_blocking = True)

        # Compute padding masks
        tgt_length = len(y[-1])
        src_pad_mask = (X == self.special_tokens["pad_token"])
        tgt_pad_mask = (y == self.special_tokens["pad_token"])
        peak_ahead_mask = self.peak_ahead_mask[:tgt_length, :tgt_length]

        # Create encoded src once
        encoded = self.encoder_stack(
            self.pos_embedder(self.embedder(X)),
            src_key_padding_mask = src_pad_mask
        )

        y_pred = self.output_linear(
            self.decoder_stack(
                self.pos_embedder(self.embedder(y)),
                memory = encoded,
                tgt_mask = peak_ahead_mask,
                tgt_key_padding_mask = tgt_pad_mask,
                memory_key_padding_mask = src_pad_mask
            )
        )

        summary_with_eos = F.pad(y[:, 1:], pad = (0,1), value = 0)
        summary_with_eos[torch.arange(y.shape[0], device = self.device), summary_with_eos.argmin(dim = 1)] = self.special_tokens["eos_token"]

        return F.cross_entropy(y_pred.view(-1, y_pred.shape[2]), summary_with_eos.view(-1))

    def predict_step(self, batch, batch_idx) -> Any:
        # Compute padding masks
        src_pad_mask = (batch == self.special_tokens["pad_token"])

        # Create encoded src once
        encoded = self.encoder_stack(
            self.pos_embedder(self.embedder(batch)),
            src_key_padding_mask = src_pad_mask
        )

        # Create target batch, with [BOS] first token 
        tgt = torch.tensor(encoded.shape[0]*[[self.special_tokens["bos_token"]]], device = next(self.parameters()).device)

        # Each target with [EOS] token is output, so it moved to preds 
        preds = []

        # If we have decoder stack, then for first decoder we can provide
        # last and all previous predicted tokens
        # but there are intermediate states between decoders
        # we have to cache, so each decoder gets last and all previous predicted from previous decoder
        cache = None

        # Generate one token at a time for each target in batch
        for _ in range(0,self.seq_len):
            # If all targets moved to preds, generating is ended
            if len(tgt) == 0: break

            # New tokens(batch_size, 1, emb_dim), previous intermediate states + new state(num_decoders, batch_size, state_length, emb_dim)
            decoder_out, cache = self.decoder_stack(
                    self.pos_embedder(self.embedder(tgt)),
                    memory = encoded,
                    memory_key_padding_mask = src_pad_mask,
                    cache = cache
            )

            # Get probability dist.(batch_size, 1, vocab_size)
            distribution = self.output_linear(decoder_out)

            # Add new tokens with highest probabilities to tgt
            tgt = torch.cat([tgt, distribution.argmax(dim = -1)], dim = 1)

            # Get indexes of targets with [EOS] token
            predicted = (tgt[:,-1] == self.special_tokens["eos_token"])

            # If there are targets with [EOS] token
            # Move targets to preds
            # Remove targets from tgt
            # Remove targets intermediate states from cache
            # Remove targets srcs
            if predicted.any():
                preds.extend(list(tgt[predicted]))
                tgt = tgt[~predicted]
                cache = cache[:, ~predicted]
                encoded = encoded[~predicted]

        # If we have targets with max length and without [EOS], they are not in preds
        if len(tgt) != 0:
            preds.extend(list(tgt))

        return preds 

    def configure_optimizers(self): 
        optimizer = Adam(
            params = self.parameters(),
            lr = 1e-6,
            betas = (0.9, 0.999),
            eps = 1e-9
        )
        return optimizer

class Trainer():
    def __init__(
            self,
            model,
            optimizer,
            loss_fn,
            scheduler,
            tokenizer,
            epoch,
            device = "cpu",
            checkpoint_path = None,
            checkpoint_by = None,
            non_blocking = False,
            warmup = False,
            max_src_len = 0,
            max_tgt_len = 0,
            batch_size = 32,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.epoch = epoch
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_by = checkpoint_by
        if self.checkpoint_by:
            self.__setattr__(self.checkpoint_by, 0)
        self.non_blocking = non_blocking
        self.warmup = warmup
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.batch_size = batch_size
    

    def train(
            self,
            train_dataloader,
            validation_dataloader = None,
            validation_rate = 1,
            model_metrics = None,
            writer = None
    ):
        self._warmup()

        print(
            f'model: {self.model.__class__.__name__}, '
            f'epochs: {self.epoch}, '
            f'batch size: {self.batch_size}, '
            f'train dataset len: {len(train_dataloader) * self.batch_size} inst.'
        )
    
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            for epoch in range(1, self.epoch + 1):
                print(f"epoch: {epoch}")
                ov_loss = 0

                self.model.train()
                for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                    loss = self._train_step(batch)
                    loss.backward()

                    ov_loss += loss 
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                self.scheduler.step()
                ov_loss = ov_loss / len(train_dataloader)
                print(f"overall loss: {ov_loss}")

                if writer:
                    writer.add_scalar(f"{self.__class__.__name__}_loss", ov_loss.type(torch.float), epoch)

                if validation_dataloader and model_metrics and epoch % validation_rate == 0:
                    self.validate(validation_dataloader, model_metrics)

                    if self.checkpoint_by:
                        new_val = model_metrics[self.checkpoint_by].compute()
                        if self.__getattribute__(self.checkpoint_by) < new_val:
                            self.__setattr__(self.checkpoint_by, new_val)
                            torch.save({
                                "model_state": self.model.state_dict(),
                                "optimizer_state": self.optimizer.state_dict(),
                                "scheduler_state": self.scheduler.state_dict(),
                                "checkpoint_by": {self.checkpoint_by: self.__getattribute__(self.checkpoint_by)}
                            }, self.checkpoint_path)

                    for name, metric in model_metrics.items():
                        if writer:
                            writer.add_scalar(f"{self.__class__.__name__}_{name}", metric.compute(), epoch)
                        metric.reset()

    def _train_step(self, batch):
        x, y = batch["document"].to(self.device, non_blocking = self.non_blocking), batch["summary"].to(self.device, non_blocking = self.non_blocking)

        y_pred = self.model(x, y)

        summary_with_eos = F.pad(y[:, 1:], pad = (0,1), value = 0)
        summary_with_eos[torch.arange(self.batch_size, device = self.device), summary_with_eos.argmin(dim = 1)] = self.tokenizer.eos_token_id

        loss = self.loss_fn(y_pred.view(-1, y_pred.shape[2]), summary_with_eos.view(-1))

        return loss

    def _warmup(self):
        if self.warmup:
            self.model.train()

            X = torch.rand(self.batch_size, self.max_src_len).type(torch.LongTensor).to(self.device)
            y = torch.rand(self.batch_size, self.max_tgt_len).type(torch.LongTensor).to(self.device)
            summary_with_eos = torch.rand(self.batch_size, self.max_tgt_len,1).type(torch.LongTensor).to(self.device)

            y_pred = self.model(X, y)
            loss = self.loss_fn(y_pred.view(-1, y_pred.shape[2]), summary_with_eos.view(-1))
            loss.backward()

            self.optimizer.zero_grad()

    def validate(self, validation_dataloader, model_metrics):
        self.model.eval(autoregressive = False)
        with torch.inference_mode():
            for batch, data in enumerate(tqdm(validation_dataloader)):
                with torch.autocast(device_type = self.device):
                    preds = self.model(data["document"].to(self.device), data["document"].to(self.device)).argmax(-1)
                preds = self.tokenizer.batch_decode(preds, skip_special_tokens = True)
                for metric in model_metrics.values():
                    metric.update(preds, self.tokenizer.batch_decode(data["summary"], skip_special_tokens = True))
            for name, metric in model_metrics.items():
                print(f'{name}: {metric.compute()}')

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.scheduler.load_state_dict(state_dict["scheduler_state"])
        for name, value in state_dict["checkpoint_by"].items():
            self.checkpoint_by = name
            self.__setattr__(self.checkpoint_by, value)
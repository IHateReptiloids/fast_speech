import torch
import torch.nn.functional as F
from tqdm import tqdm


class DefaultTrainer:
    def __init__(
        self,
        device,
        model,
        aligner,
        train_loader,
        val_loader
    ):
        raise NotImplementedError

    def train(self, num_epochs, validate=False):
        for i in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            print(f'Epoch {i + 1} train loss: {train_loss}')
            if validate:
                val_loss = self.validate()
                print(f'Epoch {i + 1} validation loss: {val_loss}')
            print('-' * 100)

    def train_batch(self, batch):
        '''
        x is of shape [bs, seq_len]
        y is of shape [bs, seq_len, n_mels]
        spec_lengths is of shape [bs]
        grapheme_lengths is of shape [bs, seq_len]
        '''
        x = batch.x.to(self.device)
        y = batch.y.to(self.device)
        assert x.shape == y.shape[:-1]
        seq_len = x.shape[-1]

        spec_lengths = batch.spec_lengths.to(self.device)
        assert spec_lengths.dtype == torch.int32
        # TODO: fix this
        grapheme_lengths = self.aligner(...)
        assert grapheme_lengths.dtype == torch.int32
        spec_lengths = grapheme_lengths.sum(dim=-1)

        mask = (torch.arange(seq_len)[None, :] < spec_lengths[:, None]) \
            .unsqueeze(-1)
        output = self.model(x, grapheme_lengths)
        loss = F.mse_loss(output * mask, y * mask)
        return loss + self.model.length_regulator.loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader):
            loss = self.train_batch(batch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        raise NotImplementedError

    @torch.no_grad()
    def validate_batch(self, batch):
        raise NotImplementedError

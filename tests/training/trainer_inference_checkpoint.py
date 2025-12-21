import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.training.trainer import QiskitTrainer


class DummyDenoiseDataset(Dataset):
    """Return (clean, noisy) with shape (1, 28, 28)."""

    def __init__(self, n: int = 20, noise_std: float = 0.1):
        self.n = n
        self.noise_std = noise_std

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        clean = torch.rand(1, 28, 28)
        noisy = clean + self.noise_std * torch.randn_like(clean)
        return clean, noisy


class SimpleConvDenoiser(nn.Module):
    """Simple model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


def make_trainer(tmp_path):
    model = SimpleConvDenoiser()
    loss_fn = nn.MSELoss()
    trainer = QiskitTrainer(model=model, loss_fn=loss_fn, metrics=None, optimizer=None, device="cpu", checkpoint_dir=tmp_path)
    return trainer


def test_predict_dataloader_respects_predict_steps(tmp_path):

    trainer = make_trainer(tmp_path)

    ds = DummyDenoiseDataset(n=20)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    out = trainer.predict(loader, predict_steps=2)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 8


def test_evaluate_respects_evaluation_steps(tmp_path):

    trainer = make_trainer(tmp_path)

    ds = DummyDenoiseDataset(n=20)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    metrics = trainer.evaluate(loader, verbose=False, evaluation_steps=1)

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)


def test_load_checkpoint_restores_model_weights(tmp_path):

    trainer = make_trainer(tmp_path)

    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.save_checkpoint(epoch=1, metrics={"loss": 0.123})
    ckpt_path = tmp_path / "checkpoint_epoch_1.pt"
    assert ckpt_path.exists()

    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(1.0)

    after_dirty = trainer.model.state_dict()
    assert any(not torch.equal(before[k], after_dirty[k]) for k in before.keys())

    trainer.load_checkpoint(ckpt_path)

    after_load = trainer.model.state_dict()
    for k in before.keys():
        assert torch.equal(before[k], after_load[k])

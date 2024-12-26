import functools

import torch
import torch.nn as nn
import yaml
from lightning import LightningModule
from sklearn.metrics import mean_absolute_error
from torch.nn import L1Loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .prithvi_backbone import MaskedAutoencoderViT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrithviRegressionModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        backbone_model, embed_dim = load_model()
        self.patch_embed = backbone_model.patch_embed
        self.blocks = backbone_model.blocks
        self.norm = backbone_model.norm
        self.final_layer = nn.Linear(embed_dim, 1)
        # self.model = nn.Sequential(*self.backbone).to(DEVICE)

        # if FREEZE_LAYER:
        #     self.freeze_layers(up_until=FREEZE_LAYER)
        # self.model = torch.compile(self.model)
        # print(self.model)
        self.loss_fn = L1Loss()
        self.lr = 1e-05

        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def freeze_layers(self, up_until=5):
        for param in self.model.parameters():
            param.requires_grad = False

        if up_until >= len(self.model.blocks):
            up_until = -1

        for param in self.model.blocks[up_until:].parameters():
            param.requires_grad = True

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    @staticmethod
    def compute_metrics(preds, targets):
        mae = mean_absolute_error(y_true=targets, y_pred=preds)

        return mae

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1) # global average pooling
        out = self.final_layer(x)
        out = torch.clip(out, min=0.0, max=1.0)
        
        return out

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = self.loss_fn(y_hat, batch[1])
        train_mae = self.compute_metrics(preds=y_hat.detach().cpu().numpy(), targets=batch[1].detach().cpu().numpy())

        self.log("train/loss", loss.item(), prog_bar=True)
        self.log("train/mae", train_mae, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean")

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = self.loss_fn(y_hat, batch[1])
        val_mae = self.compute_metrics(preds=y_hat.detach().cpu().numpy(), targets=batch[1].detach().cpu().numpy())

        self.log("eval/loss", loss, prog_bar=True)
        self.log("eval/mae", val_mae, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean")

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=500)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_model():
    checkpoint = "modelling/prithvi/Prithvi_100M.pt"
    with open("modelling/prithvi/Prithvi_100M_config.yaml", "r") as f:
        params = yaml.safe_load(f)

    # model related
    model_params = params["model_args"]
    print(model_params)
    img_size = model_params["img_size"]
    depth = model_params["depth"]
    patch_size = model_params["patch_size"]
    embed_dim = model_params["embed_dim"]
    num_heads = model_params["num_heads"]
    num_frames = model_params["num_frames"]
    tubelet_size = model_params["tubelet_size"]
    decoder_embed_dim = model_params["decoder_embed_dim"]
    decoder_num_heads = model_params["decoder_num_heads"]
    decoder_depth = model_params["decoder_depth"]

    model = MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=6,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    model.to(DEVICE)

    state_dict = torch.load(checkpoint, map_location=DEVICE)

    del state_dict["pos_embed"]
    del state_dict["decoder_pos_embed"]
    model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint from {checkpoint}")
    modules = nn.ModuleList([
        model.patch_embed,
        model.blocks,
        model.norm,
    ])

    return model, embed_dim

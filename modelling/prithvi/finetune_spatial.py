import argparse
import pandas as pd
import os
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import re
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from torch.optim import Adam
from torch.nn import L1Loss
from prithvi import MaskedAutoencoderViT
import yaml
import functools
from lightning import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch import loggers as pl_loggers
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrithviRegressionModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone, embed_dim = load_model()
        self.model = nn.Sequential(*[self.backbone, nn.Linear(embed_dim, 1)]).to(DEVICE)

        # if FREEZE_LAYER:
        #     self.freeze_layers(up_until=FREEZE_LAYER)
        self.model = torch.compile(self.model)
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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch[0])
        loss = self.loss_fn(y_hat, batch[1])
        train_mae = self.compute_metrics(preds=y_hat, targets=batch[1])

        # Log the train loss to Tensorboard
        self.log("train/loss", loss.item(), prog_bar=True)
        self.log("train/mae", train_mae, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean")

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch[0])
        loss = self.loss_fn(y_hat, batch[1])
        val_mae = self.compute_metrics(preds=y_hat, targets=batch[1])

        self.log("eval/loss", loss, prog_bar=True)
        self.log("eval/mae", val_mae, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean")

        return loss

    # def on_train_epoch_end(self):
    #     train_all_outputs = torch.stack(self.train_step_outputs)
    #     train_all_targets = torch.stack(self.train_step_targets)

    #     self.train_step_outputs.clear()
    #     self.train_step_targets.clear()

    # def on_validation_epoch_end(self):
    #     val_all_outputs = torch.stack(self.val_step_outputs)
    #     val_all_targets = torch.stack(self.val_step_targets)


    #     self.val_step_outputs.clear()
    #     self.val_step_targets.clear()

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
    model = nn.Sequential(*[model.patch_embed, model.blocks, model.norm])
    
    return model, embed_dim



def main(fold, imagery_path, batch_size, normalize):
    normalization = 30000.
    imagery_size = 336

    data_folder = r'survey_processing/processed_data'

    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}.csv')

    available_imagery = []
    for d in os.listdir(imagery_path):
        if d[-2] == "L":
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))

    def is_available(centroid_id):
        for centroid in available_imagery:
            if centroid_id in centroid:
                return True
        return False
    train_df = train_df[train_df['CENTROID_ID'].apply(is_available)]
    test_df = test_df[test_df['CENTROID_ID'].apply(is_available)]

    def filter_contains(query):
        """
        Returns a list of items that contain the given query substring.
        
        Parameters:
            items (list of str): The list of strings to search within.
            query (str): The substring to search for in each item of the list.
            
        Returns:
            list of str: A list containing all items that have the query substring.
        """
        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)
    
    predict_target = ["deprived_sev"]

    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [c for c in train_df.columns if c == col or re.match(f"^{col}_[^a-zA-Z]", c)]
        )
    # Drop rows with NaN values in the filtered subset of columns
    train_df = train_df.dropna(subset=filtered_predict_target)
    predict_target = sorted(filtered_predict_target)

    def load_and_preprocess_image(path, grouped_bands=[2, 3, 4, 5, 6, 7]):
        with rasterio.open(path) as src:
            b1 = src.read(grouped_bands[0])
            b2 = src.read(grouped_bands[1])
            b3 = src.read(grouped_bands[2])
            b4 = src.read(grouped_bands[3])
            b5 = src.read(grouped_bands[4])
            b6 = src.read(grouped_bands[5])

            # Stack and normalize the bands
            img = np.dstack((b1, b2, b3, b4, b5, b6))
            img = img / normalization  # Normalize to [0, 1] (if required)

        img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
        img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range

        # Scale back to [0, 255] for visualization purposes
        # img = (img * 255).astype(np.uint8)

        return img

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set your desired seed
    seed = 42
    set_seed(seed)
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)

    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(1).float()

            print("=================SHAPEEEEEE=========", image_tensor.shape)
            target = torch.tensor(item[predict_target], dtype=torch.float32)
            return image_tensor, target

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imagery_size, imagery_size)),
            transforms.Normalize(
                mean=[
                    0.033349706741586264,
                    0.05701185520536176,
                    0.05889748132001316,
                    0.2323245113436119,
                    0.1972854853760658,
                    0.11944914225186566,
                ],
                std=[
                    0.02269135568823774,
                    0.026807560223070237,
                    0.04004109844362779,
                    0.07791732423672691,
                    0.08708738838140137,
                    0.07241979477437814,
                ]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imagery_size, imagery_size)),
        ])

    train_dataset = CustomDataset(train, transform)
    val_dataset = CustomDataset(validation, transform)

    class LandsatDataset(LightningDataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 1
        
        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    data_module = LandsatDataset()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    num_train_batches = (len(train_dataloader) // batch_size)

    regression_module = PrithviRegressionModel()

    model_save_name = f'modelling/prithvi/model/Prithvi100M_{fold}/'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=model_save_name)
    model_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("models", model_save_name), save_top_k=1, save_last=True, 
                                                monitor="eval/mae", mode="min")
    early_stopping_callback = EarlyStopping(monitor="eval/mae", patience=7, mode="min")
    # batch_size_finder = BatchSizeFinder(mode="power", init_val=1)

    trainer = Trainer(
        logger=tb_logger,
        accumulate_grad_batches=4,
        check_val_every_n_epoch=1,
        val_check_interval=1,
        limit_train_batches=num_train_batches,
        enable_progress_bar=True,
        # callbacks=[model_checkpoint_callback, early_stopping_callback, batch_size_finder]
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )
    tuner = Tuner(trainer=trainer)
    tuner.lr_find(model=regression_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.fit(model=regression_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
    # base_model = torch.hub.load('facebookresearch/dinov2', model_name)

    # def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss
    #     }, filename)

    # torch.cuda.empty_cache()
    # class ViTForRegression(nn.Module):
    #     def __init__(self, base_model):
    #         super().__init__()
    #         self.base_model = base_model
    #         # Assuming the original model outputs 768 features from the transformer
    #         self.regression_head = nn.Linear(emb_size, len(predict_target))  # Output one continuous variable

    #     def forward(self, pixel_values):
    #         outputs = self.base_model(pixel_values)
    #         # We use the last hidden state
    #         return torch.sigmoid(self.regression_head(outputs))

    
    # print(f"Using {DEVICE}")
    # model = ViTForRegression(base_model).to(DEVICE)
    # best_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_best_.pth'
    # last_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_last_.pth'
    # if os.path.exists(last_model):
    #     last_state_dict = torch.load(last_model)
    #     best_error = torch.load(best_model)['loss']
    #     epochs_ran = last_state_dict['epoch']
    #     model.load_state_dict(last_state_dict['model_state_dict'])
    #     print('Found existing model')
    # else:
    #     epochs_ran = 0
    #     best_error = np.inf

    # # Move model to appropriate device
    # model.to(DEVICE)

    # base_model_params = {'params': model.base_model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    # head_params = {'params': model.regression_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}

    # # Setup the optimizer
    # optimizer = torch.optim.Adam([base_model_params, head_params])
    # loss_fn = L1Loss()

    # for epoch in range(epochs_ran+1, num_epochs):
    #     torch.cuda.empty_cache()
    #     model.train()
    #     print('Training...')
    #     for batch in tqdm(train_loader):
    #         images, targets = batch
    #         images, targets = images.to(device), targets.to(device)
            
    #         # Forward pass
    #         outputs = model(images)
    #         loss = loss_fn(outputs, targets)
            
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     torch.cuda.empty_cache()
    #     # Validation phase
    #     model.eval()
    #     val_loss = []
    #     indiv_loss = []
    #     print('Validating...')
    #     for batch in val_loader:
    #         images, targets = batch
    #         images, targets = images.to(device), targets.to(device)
            
    #         # Forward pass
    #         with torch.no_grad():
    #             outputs = model(images)
    #         batch_loss = loss_fn(outputs, targets)
    #         val_loss.append(batch_loss.item())
    #         indiv_loss.append(torch.mean(torch.abs(outputs-targets), axis=0))
        
    #     # Compute mean validation loss
    #     mean_val_loss = np.mean(val_loss)   
    #     mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

    #     if mean_val_loss< best_error:
    #         save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
    #         best_error = mean_val_loss
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
    #     save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=str, help='CV fold')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    main(fold=args.fold, imagery_path=args.imagery_path, batch_size=args.batch_size, normalize=args.normalize)
    
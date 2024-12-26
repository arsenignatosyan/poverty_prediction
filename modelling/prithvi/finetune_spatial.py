import argparse
import os
import random
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as transforms
from lightning import LightningDataModule, Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, BatchSizeFinder
from lightning.pytorch.tuner.tuning import Tuner
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

from .prithvi_regressor import PrithviRegressionModel


def main(fold, imagery_path, batch_size, normalize):
    normalization = 30000.
    imagery_size = 224

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
            self.batch_size = batch_size

        def train_dataloader(self):
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    data_module = LandsatDataset()
    # train_dataloader = data_module.train_dataloader()
    # val_dataloader = data_module.val_dataloader()
    regression_module = PrithviRegressionModel()

    model_save_name = f'modelling/prithvi/model/Prithvi100M_{fold}/'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", name=model_save_name)
    model_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("models", model_save_name), save_top_k=1,
                                                save_last=True,
                                                monitor="eval/mae", mode="min")
    early_stopping_callback = EarlyStopping(monitor="eval/mae", patience=7, mode="min")
    batch_size_finder = BatchSizeFinder(mode="power", init_val=1)
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=tb_logger,
        accumulate_grad_batches=4,
        val_check_interval=1.0,
        enable_progress_bar=True,
        callbacks=[model_checkpoint_callback, early_stopping_callback, batch_size_finder]
    )
    tuner = Tuner(trainer=trainer)
    tuner.lr_find(model=regression_module, datamodule=data_module, num_training=20)
    trainer.fit(model=regression_module, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=str, help='CV fold')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    main(fold=args.fold, imagery_path=args.imagery_path, batch_size=args.batch_size, normalize=args.normalize)

import os
from model import Network
from data import TaggingDataset, collate_fn, collate_fn_test, get_train_probs, load_data
from utils import train_epoch, predict, predict_train
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import set_seed
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")
import copy 
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import yaml
import hydra

import logging
log = logging.getLogger(__name__)


def get_config():
    try:
        path = os.environ["HYDRA_CONFIG_PATH"]
        name = os.environ["HYDRA_CONFIG_NAME"]
    except:
        path = os.path.dirname(os.environ["HYDRA_CONFIG_PATH"])
        name = os.path.basename(os.environ["HYDRA_CONFIG_PATH"])
    return path, name


def run_train_model(config, workdir):
    CUDA_DEV = 0
    NUM_TAGS = 256
    set_seed(config.seed)
    
    df_train, df_test, track_idx2embeds = load_data()
    dict_tags = get_train_probs(df_train)
    
    test_dataset = TaggingDataset(df_test, track_idx2embeds=track_idx2embeds, testing=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_test)
    
    max_length = config.data.max_length
    early_stopping = config.training.early_stopping if "early_stopping" in config.training.keys() else True
    lstm = config.model.lstm if "lstm" in config.model.keys() else False
    
    if config.data.k_folds == 1:
        model = Network(num_classes=NUM_TAGS, nhead=config.model.nhead, num_layers=config.model.num_layers).to(CUDA_DEV)
        criterion = nn.BCEWithLogitsLoss().to(CUDA_DEV)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.training.T_0, eta_min=config.training.eta_min)

        train_dataset = TaggingDataset(df_train, track_idx2embeds=track_idx2embeds, aug=config.data.aug)
        val_dataset = TaggingDataset(df_train[-1000:], track_idx2embeds=track_idx2embeds)
        
        train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
            
        for epoch in range(config.training.epochs):
            train_epoch(model, train_dataloader, criterion, optimizer, scheduler, max_length=max_length, print_loss=True, iteration_step=700, epoch=epoch)
            track_idxs, predictions, targets = predict_train(model, val_dataloader, max_length=max_length)
            ap = sklearn.metrics.average_precision_score(targets, predictions)
            if epoch in [60, 68, 70, 75]:
                
                track_idxs, predictions = predict(model, test_dataloader, max_length=max_length)
                for i, c in enumerate(predictions.argmax(-1)):
                    probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
                    probs[c] = 2
                    predictions[i] = predictions[i] * probs
                    predictions[i] /= predictions[i].sum()

                predictions_df = pd.DataFrame([
                    {'track': track, 'prediction': ','.join([str(p) for p in probs])}
                    for track, probs in zip(track_idxs, predictions)
                ])
                predictions_df.to_csv(f'{workdir}/prediction_full_{epoch}.csv', index=False)
                torch.save(model.state_dict(), f'{workdir}/models_full_{epoch}.pt')
                
            log.info(f"epoch: {epoch}, AP: {ap}")
            
        track_idxs, predictions = predict(model, test_dataloader, max_length=max_length)

        for i, c in enumerate(predictions.argmax(-1)):
            probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
            probs[c] = 2
            predictions[i] = predictions[i] * probs
            predictions[i] /= predictions[i].sum()

        predictions_df = pd.DataFrame([
            {'track': track, 'prediction': ','.join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ])
        predictions_df.to_csv(f'{workdir}/prediction_full.csv', index=False)
        torch.save(model.state_dict(), f'{workdir}/models_full.pt')
    else:
        kf = KFold(n_splits=config.data.k_folds, random_state=config.seed, shuffle=True)
        best_preds = []
        best_preds_last = []
        fold_ap = []
        for fold_i, (train_index, test_index) in enumerate(kf.split(df_train)):
            model = Network(num_classes=NUM_TAGS, nhead=config.model.nhead, num_layers=config.model.num_layers, lstm=lstm).to(CUDA_DEV)
            criterion = nn.BCEWithLogitsLoss().to(CUDA_DEV)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.training.T_0, eta_min=config.training.eta_min)
    
            ls = config.label_smoothing if "label_smoothing" in config.keys() else 0
            train_dataset = TaggingDataset(df_train.iloc[train_index], track_idx2embeds=track_idx2embeds, aug=config.data.aug, label_smoothing=ls)
            val_dataset = TaggingDataset(df_train.iloc[test_index], track_idx2embeds=track_idx2embeds)

            train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, collate_fn=collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

            best_ap = 0
            best_model = None
            best_preds_fold = None
            best_epoch = 0
            
            for epoch in range(config.training.epochs):
                train_epoch(model, train_dataloader, criterion, optimizer, scheduler, max_length=max_length, print_loss=True, iteration_step=600, epoch=epoch)
                track_idxs, predictions, targets = predict_train(model, val_dataloader, max_length=max_length)
                ap = sklearn.metrics.average_precision_score(targets, predictions)
                if (ap > best_ap):
                    log.info(f"Fold: {fold_i}, epoch: {epoch}, AP: {ap} --------- NEW BEST")
                    best_ap = ap
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                else:
                    log.info(f"Fold: {fold_i}, epoch: {epoch}, AP: {ap}")
                if ((epoch - best_epoch) > 15) and early_stopping:
                    log.info(f"Fold: {fold_i}, early stopped at epoch: {epoch}, since 15 epochs no improvements by AP")
                    break
                    
            if not early_stopping:
                track_idxs, best_preds_fold = predict(model, test_dataloader, max_length=max_length)
                best_preds_last.append(best_preds_fold)
                predictions = copy.deepcopy(best_preds_fold)
                fold_ap.append(best_ap)

                for i, c in enumerate(predictions.argmax(-1)):
                    probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
                    probs[c] = 2
                    predictions[i] = predictions[i] * probs
                    predictions[i] /= predictions[i].sum()

                predictions_df = pd.DataFrame([
                    {'track': track, 'prediction': ','.join([str(p) for p in probs])}
                    for track, probs in zip(track_idxs, predictions)
                ])
                predictions_df.to_csv(f'{workdir}/last_prediction_{fold_i}.csv', index=False)
                torch.save(model.state_dict(), f'{workdir}/last_model_{fold_i}.pt')

            track_idxs, best_preds_fold = predict(best_model, test_dataloader, max_length=max_length)
            best_preds.append(best_preds_fold)
            predictions = copy.deepcopy(best_preds_fold)
            fold_ap.append(best_ap)
            
            for i, c in enumerate(predictions.argmax(-1)):
                probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
                probs[c] = 2
                predictions[i] = predictions[i] * probs
                predictions[i] /= predictions[i].sum()

            predictions_df = pd.DataFrame([
                {'track': track, 'prediction': ','.join([str(p) for p in probs])}
                for track, probs in zip(track_idxs, predictions)
            ])
            predictions_df.to_csv(f'{workdir}/prediction_{fold_i}.csv', index=False)
            torch.save(best_model.state_dict(), f'{workdir}/model_{fold_i}.pt')
            
        log.info(f"Folds AP: {fold_ap}, Mean: {np.mean(fold_ap)}")
        predictions = np.mean(best_preds, axis=0)
        for i, c in enumerate(predictions.argmax(-1)):
            probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
            probs[c] = 2
            predictions[i] = predictions[i] * probs
            predictions[i] /= predictions[i].sum()

        predictions_df = pd.DataFrame([
            {'track': track, 'prediction': ','.join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ])
        predictions_df.to_csv(f'{workdir}/prediction_mean_folds.csv', index=False)
        
        
        predictions = np.mean(best_preds_last, axis=0)
        for i, c in enumerate(predictions.argmax(-1)):
            probs = np.array([1 + dict_tags[c].get(t, 0) for t in np.arange(predictions.shape[1])])
            probs[c] = 2
            predictions[i] = predictions[i] * probs
            predictions[i] /= predictions[i].sum()

        predictions_df = pd.DataFrame([
            {'track': track, 'prediction': ','.join([str(p) for p in probs])}
            for track, probs in zip(track_idxs, predictions)
        ])
        predictions_df.to_csv(f'{workdir}/prediction_last_mean_folds.csv', index=False)
            

@hydra.main(
    config_path=get_config()[0],
    config_name=get_config()[1],
)
def main(config):
    os.environ["WANDB_WATCH"] = "False"  # To disable Huggingface logging
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())
    run_train_model(config, auto_generated_dir)


if __name__ == "__main__":
    main()
import torch
from datetime import datetime
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

CUDA_DEV = 0

def train_epoch(model, loader, criterion, optimizer, scheduler, max_length=64, print_loss=True, iteration_step=100, epoch=0):
    model.train()
    running_loss = None
    alpha = 0.8
    iters = len(loader)
    for iteration,data in enumerate(loader):
        optimizer.zero_grad()
        track_idxs, embeds, target = data
        embeds = [x.to(CUDA_DEV) for x in embeds]
        embeds = pad_sequence(embeds, padding_value=-1, batch_first=True)[:, :max_length, :]
        target = target.to(CUDA_DEV)
        pred_logits = model(embeds)
        pred_probs = torch.sigmoid(pred_logits)
        ce_loss = criterion(pred_logits, target)
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step(epoch + iteration / iters)
        
        if running_loss is None:
            running_loss = ce_loss.item()
        else:
            running_loss = alpha * running_loss + (1 - alpha) * ce_loss.item()
        if (iteration % iteration_step == 0) and print_loss:
            print('   {} batch {} running loss {} loss {}'.format(
                datetime.now(), iteration + 1, running_loss, ce_loss.item()
            ))
            
def predict(model, loader, max_length=64):
    model.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            track_idx, embeds = data
            embeds = [x.to(CUDA_DEV) for x in embeds]
            embeds = pad_sequence(embeds, padding_value=-1, batch_first=True)[:, :max_length, :]
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)
            predictions.append(pred_probs.cpu().numpy())
            track_idxs.append(track_idx.numpy())
    predictions = np.vstack(predictions)
    track_idxs = np.vstack(track_idxs).ravel()
    return track_idxs, predictions


def predict_train(model, loader, max_length=64):
    model.eval()
    track_idxs = []
    predictions = []
    targets = []
    with torch.no_grad():
        for data in loader:
            track_idx, embeds, target = data
            embeds = [x.to(CUDA_DEV) for x in embeds]
            embeds = pad_sequence(embeds, padding_value=-1, batch_first=True)[:, :max_length, :]
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)
            predictions.append(pred_probs.cpu().numpy())
            track_idxs.append(track_idx.numpy())
            targets.append(target.numpy())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    track_idxs = np.vstack(track_idxs).ravel()
    return track_idxs, predictions, targets
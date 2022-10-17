import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import argparse

from model import ResNet
from data import build_dataloaders
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/Kannada_mnist_experiment_1')

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

class Trainer():
    def __init__(
        self, 
        learning_rate,
        weight_decay,
        trained_model = None, 
        **model_kwargs
        ):
        print(model_kwargs)
        if not trained_model:
            self.model = ResNet(**model_kwargs)
        else:
            self.model = trained_model
        self.dataloaders = build_dataloaders()
        self.loss_module = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_step(self, data):
        labels, features = data['label'], data['features']
        ## Step 1: Move input data to device (only strictly necessary if we use GPU)
        features = features.to(device)
        labels = labels.to(device)
        preds = self.model(features)
        loss = self.loss_module(preds, labels)
        accuracy = (preds.argmax(dim=-1) == labels).float().mean()
        self.optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        ## Step 5: Update the parameters
        self.optimizer.step()
        return loss, accuracy

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        dataloader = self.dataloaders['train']
        for data in dataloader:
            loss, accuracy = self.train_step(data)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        return running_loss/len(dataloader), running_accuracy/len(dataloader)

    def eval_model(self):
        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        dataloader = self.dataloaders['val']
        with torch.no_grad():
            for data in dataloader:
                labels, features = data['label'], data['features']
                features = features.to(device)
                labels = labels.to(device)
                preds = self.model(features)
                loss = self.loss_module(preds, labels)
                accuracy = (preds.argmax(dim=-1) == labels).float().mean()
                running_loss += loss.item()
                running_accuracy += accuracy.item()

            return running_loss/len(dataloader), running_accuracy/len(dataloader)
    
    def predict(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        results = pd.DataFrame()
        with torch.no_grad():
            for data in dataloader:
                id, features = data['label'], data['features']
                features = features.to(device)
                predictions = self.model(features).argmax(dim=-1)
                results.write(id, predictions)
        return results


def write_predictions(pandas_df, file_path):
    pandas_df.write_csv(file_path)


def train_model(args, **model_kwargs):

    trainer = Trainer(args)

    best_loss = np.float('inf')

    for epoch in tqdm(range(args.num_epochs)):
        train_loss, train_acc = trainer.train_epoch()
        # log the running loss and accuracy
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = trainer.eval_model(mode='val')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save_dict(trainer.model, 'best_model.h5')

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

    best_model = torch.load_dict('best_model.h5')

    best_model_trainer = Trainer(best_model)

    predictions = best_model_trainer.predict()

    write_predictions(predictions)



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('num_epochs', default=50)
    args.add_argument('lr', default=1e-5)
    args.add_argument('weight_decay', default=1e-5)

    train_model(args)
        





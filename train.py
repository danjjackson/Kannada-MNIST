import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from copy import deepcopy
import pandas as pd
import numpy as np
from datetime import datetime

from utils import create_model
from data import build_dataloaders

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Trainer():
    def __init__(
        self, 
        num_epochs,
        model_name,
        batch_size,
        optimizer_name,
        optimizer_hparams,
        model_hparams):

        self.num_epochs = num_epochs
        self.model = create_model(model_name, model_hparams)
        self.model = self.model.to(device)
        self.dataloaders = build_dataloaders(batch_size)
        self.loss_module = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.configure_optimizers()

    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            self.optimizer = optim.AdamW(
                self.model.parameters(), **self.optimizer_hparams)
        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(), **self.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.optimizer_name}\""

        self.exp_lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.num_epochs // 4, 
            gamma=0.1
            )


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        train_loader = self.dataloaders['train']
        for labels, features in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            preds = self.model(features)
            loss = self.loss_module(preds, labels)
            total_loss += loss.item()
            correct += (preds.argmax(dim=-1) == labels).sum()
            
            loss.backward()

            self.optimizer.step()
        
        epoch_loss = total_loss/len(train_loader.dataset)
        accuracy = correct/len(train_loader.dataset)

        print('Epoch {}/{}\nTrain loss: {:.4f}, Train accuracy: {}/{} ({:.3f}%)\n'.format(
            epoch+1, self.num_epochs, epoch_loss, correct, len(train_loader.dataset), 100. * accuracy)) 
        
        self.exp_lr_scheduler.step()

        return epoch_loss, accuracy
            

    def eval_model(self, mode='val'):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        dataloader = self.dataloaders[mode]
        with torch.no_grad():
            for labels, features in dataloader:
                features = features.to(device)
                labels = labels.to(device)
                preds = self.model(features)
                loss = self.loss_module(preds, labels)
                total_loss += loss.item()
                correct += (preds.argmax(dim=-1) == labels).sum()

        epoch_loss = total_loss/len(dataloader.dataset)
        accuracy = correct/len(dataloader.dataset)

        print('Val loss: {:.4f}, Val accuracy: {}/{} ({:.3f}%)\n'.format(
            epoch_loss, correct, len(dataloader.dataset), 100. * accuracy))  

        return epoch_loss, accuracy
    
    def predict(self):
        self.model.eval()

        dataloader = self.dataloaders['test']
        results = pd.DataFrame()
        with torch.no_grad():
            for id, features in dataloader:
                id = id.cpu().numpy()
                features = features.to(device)
                preds = self.model(features)
                predictions = preds.argmax(dim=-1).cpu().numpy()
                result = np.vstack((id, predictions))
                result = pd.DataFrame(result.T)
                results = pd.concat([results, result])
        return results

def train_model(config, num_epochs):

    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    writer = SummaryWriter(f'runs/{config.model_name}_{dt_string}')

    trainer = Trainer(num_epochs, **config)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        # log the running loss and accuracy
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = trainer.eval_model()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = deepcopy(trainer.model.state_dict())
            torch.save(best_model_state, 'best_model.pt')

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

    
    trainer.model.load_state_dict(torch.load('best_model.pt'))

    results = trainer.predict()
    results.to_csv('submissions.csv', header = ['id', 'label'], index=False)
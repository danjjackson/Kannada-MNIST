import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet
from data import build_dataloaders

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/Kannada_mnist_experiment_1')

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

class Trainer():
    def __init__(self, **model_kwargs) -> None:
        self.model = ResNet(model_kwargs)
        self.dataloaders = build_dataloaders()
        self.loss_module = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters())

    def train_step(self, data):
        labels, features = data['label'], data['data']
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
        dataloader = self.dataloader['train']
        for data in dataloader:
            loss, accuracy = self.train_step(data)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        return running_loss/len(dataloader), running_accuracy/len(dataloader)

    def eval_model(self, mode='val'):
        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        dataloader = self.dataloaders[mode]
        with torch.no_grad():
            for data in dataloader:
                labels, features = data['label'], data['data']
                features = features.to(device)
                labels = labels.to(device)
                preds = self.model(features)
                loss = self.loss_module(preds, labels)
                accuracy = (preds.argmax(dim=-1) == labels).float().mean()
                running_loss += loss.item()
                running_accuracy += accuracy.item()

            return running_loss/len(dataloader), running_accuracy/len(dataloader)

def train_model(config, num_epochs, **model_kwargs):

    trainer = Trainer(model_kwargs)

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch()
        # log the running loss and accuracy
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        
        val_loss, val_acc = trainer.eval_model(mode='val')
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)

    test_loss, test_acc = trainer.eval_model(mode='test')

        





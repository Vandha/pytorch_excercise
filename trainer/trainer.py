import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import itertools
import copy


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epochs: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def run_experiment(self):
        best_acc = 0.0
        best_state_dict = None
        for optimizer_cls in [torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad]:
            for lr in [0.01, 0.001, 0.0001]:
                optimizer = optimizer_cls(self.model.parameters(), lr=lr)
                for weight_decay in [0.0, 0.001, 0.01]:
                    for momentum in [0.0, 0.9]:
                        config = {'optimizer': str(optimizer_cls.__name__), 'lr': lr, 'weight_decay': weight_decay,
                                'momentum': momentum}
                        self.set_config(config)

                        # create new optimizer and lr scheduler with current config
                        self.optimizer = optimizer
                        self.lr_scheduler = self.get_lr_scheduler()

                        # train the model
                        for epoch in range(1, self.epochs + 1):
                            result = self._train_epoch(epoch)
                            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch, self.epochs,
                                                                                            result['loss'])
                            for metric in self.metric_ftns:
                                message += ' {}: {:.4f}'.format(metric.__name__, result[metric.__name__])
                            self.logger.info(message)

                            if self.do_validation:
                                val_result = self._valid_epoch(epoch)
                                message = 'Epoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch, self.epochs,
                                                                                                        val_result['loss'])
                                for metric in self.metric_ftns:
                                    message += ' {}: {:.4f}'.format(metric.__name__, val_result[metric.__name__])
                                self.logger.info(message)

                            # update tensorboard and learning rate
                            if self.lr_scheduler is not None:
                                lr = self.lr_scheduler.get_lr()[0]
                                self.writer.add_scalar('learning_rate', lr, epoch)
                                self.logger.debug('Learning rate: {}'.format(lr))
                            self.writer.set_step(epoch)

                        # check if current hyperparameters gave best accuracy
                        if self.do_validation:
                            acc = val_result['accuracy']
                            if acc > best_acc:
                                best_acc = acc
                                best_state_dict = self.model.state_dict()
                                self.logger.info('Found new best accuracy: {:.4f}'.format(best_acc))
                                self.logger.info('Saving state dict to {}'.format(self.config['checkpoint']))
                                torch.save(best_state_dict, self.config['checkpoint'])

        self.logger.info('Best accuracy: {:.4f}'.format(best_acc))
        self.model.load_state_dict(best_state_dict)
        return self.model

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class ExperimentTrainer(Trainer):
    """
    Trainer class for conducting hyperparameter experiments
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, data_loader,
                         valid_data_loader, lr_scheduler, len_epoch)
        self.hyperparams = config['hyperparams'] 
        self.results = {} 
    def run_experiment(self):
        """
        Run the hyperparameter experiment
        """
        for hp in self.hyperparams.keys():
            values = self.hyperparams[hp]
            print(f'Testing hyperparameter {hp} with values: {values}')
            results = []
            for combination in itertools.product(*values):
                params = copy.deepcopy(self.config) 
                for i, val in enumerate(combination):
                    params[hp][i] = val 
                self.config = params 
                result = self._train() 
                results.append((combination, result)) 
            self.results[hp] = results 

    def _train(self):
        """
        Train the model with the current configuration and return the validation accuracy
        """
        best_val_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)

            self.logger.info(f'Train Epoch: {epoch} {train_log}')
            self.logger.info(f'Validation Epoch: {epoch} {val_log}')

            val_acc = val_log['accuracy']
            if val_acc > best_val_acc:
                self._save_checkpoint(epoch, save_best=True)
                best_val_acc = val_acc

        return best_val_acc
import numpy as np
import torch
from base import BaseTrainer
import math
import time


class DCRNNTrainer(BaseTrainer):
    """
    DCRNN trainer class
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, val_len_epoch=None):
        super(DCRNNTrainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len_epoch
        self.val_len_epoch = val_len_epoch
        self.cl_decay_steps = config["trainer"]["cl_decay_steps"]

        self.max_grad_norm = config["trainer"]["max_grad_norm"]
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(20)
        # self.log_step = int(np.sqrt(data_loader.batch_size))  # sqrt(128)  sqrt(64)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader.get_iterator()):
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target)
            label = target[..., :self.model.output_dim]  # (..., 1)  supposed to be numpy array
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.len_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.cl_decay_steps)

            output = self.model(data, target, teacher_forcing_ratio)
            output = torch.transpose(output.view(12, self.model.batch_size, self.model.num_nodes,
                                                 self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)

            loss = self.loss(output.cpu(), label)  # loss is self-defined, need cpu input
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        log.update({'Time': "{:.4f}s".format(training_time)})
        return log, training_time

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader.get_iterator()):
                data = torch.FloatTensor(data)
                target = torch.FloatTensor(target)
                label = target[..., :self.model.output_dim]  # (..., 1)  supposed to be numpy array
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data, target, 0)
                output = torch.transpose(output.view(12, self.model.batch_size, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)  # back to (50, 12, 207, 1)

                loss = self.loss(output.cpu(), label)

                self.writer.set_step((epoch - 1) * self.val_len_epoch + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / self.val_len_epoch,
            'val_metrics': (total_val_metrics / self.val_len_epoch).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))

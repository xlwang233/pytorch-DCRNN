import numpy as np
import torch
import torch.optim as optim

from lib import utils
from lib import metrics
from lib.utils import load_graph_data, count_parameters
from lib.metrics import masked_mae_loss
from model.dcrnn_model import DCRNNModel
# from model.dcrnn_supervisor import DCRNNSupervisor

import time
import math
from tqdm import tqdm
import yaml
import argparse
import collections
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser



def train(model, train_loader, epoch, optimizer, criterion, clip):
    """
    progress bar code reference:
    https://gist.github.com/harrisonpim/35b55da91103f76a67053d6328db7ce5
    """
    model.train()
    epoch_loss = 0
    cnt = 0

    loop = tqdm(enumerate(train_loader.get_iterator()), total=num_train_iteration_per_epoch)
    # for _, (x, y) in enumerate(train_loader):
    for _, (x, y) in loop:
        # x/y shape (50, 12, 207, 2)
        # convert data to pytorch tensors
        cnt += 1
        x = torch.FloatTensor(x).cuda()
        y = torch.FloatTensor(y).cuda()
        optimizer.zero_grad()
        outputs = model(x, y)  # (seq_length+1, batch_size, num_nodes*output_dim)  (13, 50, 207*1)
        outputs = torch.transpose(outputs[1:].view(12, batch_size, num_nodes, output_dim), 0, 1)  # back to (50, 12, 207, 1)
        labels = y[..., :output_dim]  # (..., 1)
        loss = criterion(outputs.cpu(), labels.cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_description('Epoch {}/{}'.format(epoch + 1, epochs))
        loop.set_postfix(train_loss=loss.item())
    return epoch_loss / cnt


def evaluate(model, val_loader, epoch, criterion):
    model.eval()
    epoch_loss = 0
    cnt = 0
    loop = tqdm(enumerate(val_loader.get_iterator()), total=num_val_iteration_per_epoch)
    with torch.no_grad():
        for i, (x, y) in loop:
            cnt += 1
            x = torch.FloatTensor(x).cuda()
            y = torch.FloatTensor(y).cuda()
            outputs = model(x, y, 0)  # (seq_length+1, batch_size, num_nodes*output_dim)  (13, 50, 207*1)
            outputs = torch.transpose(outputs[1:].view(12, batch_size, num_nodes, output_dim), 0, 1)  # back to (50, 12, 207, 1)
            labels = y[..., :output_dim]  # (..., 1)
            # outputs = outputs.detach()
            # labels = labels.detach()
            loss = criterion(outputs.cpu(), labels.cpu())
            epoch_loss += loss.item()
            loop.set_description('Epoch {}/{}'.format(epoch + 1, epochs))
            loop.set_postfix(val_loss=loss.item())

    return epoch_loss / cnt


def test(model, test_loader, scaler):
    model.eval()
    y_preds = torch.FloatTensor([])
    y_truths = data['y_test']  # (6850, 12, 207, 2)
    y_truths = scaler.inverse_transform(y_truths)
    predictions = []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader.get_iterator()):
            x = torch.FloatTensor(x).cuda()
            y = torch.FloatTensor(y).cuda()
            outputs = model(x, y, 0)  # (seq_length+1, batch_size, num_nodes*output_dim)  (13, 50, 207*1)
            y_preds = torch.cat([y_preds, outputs], dim=1)
    y_preds = torch.transpose(y_preds, 0, 1)
    y_preds = y_preds.detach().numpy()  # cast to numpy array
    print("--------test results--------")
    for horizon_i in range(y_truths.shape[1]):
        y_truth = y_truths[:, horizon_i, :, 0]

        y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i, :, 0])
        predictions.append(y_pred)

        mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
        mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
        print(
            "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                horizon_i + 1, mae, mape, rmse
            )
        )
    outputs = {
        'predictions': predictions,
        'groundtruth': y_truths
    }
    return outputs

if __name__ == '__main__':
    # Parameter setting
    # Data parameters
    batch_size = 50
    graph_pkl_filename = 'data/sensor_graph/adj_mx_unix.pkl'
    # Model parameters
    horizon = 12
    input_dim = 2
    l1_decay = 0
    max_diffusion_step = 2
    num_nodes = 207
    num_rnn_layers = 2
    output_dim = 1
    rnn_units = 64
    seq_len = 12
    use_curriculum_learning = True
    # Training parameters
    epochs = 1
    test_every_n_epochs = 10
    patience = 50
    max_grad_norm = 5

    # Prepare data
    graph_pkl_filename = 'data/sensor_graph/adj_mx_unix.pkl'
    _, _, adj_mat = load_graph_data(graph_pkl_filename)

    data = utils.load_dataset(dataset_dir='data/METR-LA',
                              batch_size=batch_size,
                              test_batch_size=batch_size)
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print((k, v.shape))

    num_train_sample = data['x_train'].shape[0]
    num_val_sample = data['x_val'].shape[0]

    # get number of iterations per epoch for progress bar
    num_train_iteration_per_epoch = math.ceil(num_train_sample / batch_size)
    num_val_iteration_per_epoch = math.ceil(num_val_sample / batch_size)

    scaler = data['scaler']

    train_data_loader = data['train_loader']
    val_data_loader = data['val_loader']
    test_data_loader = data['test_loader']

    # Initialize model
    model = DCRNNModel(batch_size=batch_size, enc_input_dim=input_dim, dec_input_dim=output_dim,
                       adj_mat=adj_mat, max_diffusion_step=max_diffusion_step,
                       num_nodes=num_nodes, num_rnn_layers=num_rnn_layers,
                       rnn_units=rnn_units, seq_len=seq_len, output_dim=output_dim)
    # Count number of trainable parameters
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # A GPU should be available
    model = model.cuda()
    print(model)
    # Loss function
    null_val = 0.
    loss_fn = masked_mae_loss(scaler, null_val)  # return a masked loss function

    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # train
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_data_loader, epoch, optimizer, loss_fn, max_grad_norm)
        valid_loss = evaluate(model, val_data_loader, epoch, loss_fn)

        end_time = time.time()

        epoch_secs = end_time - start_time

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'output/models/model.pkl')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_secs:.6}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    # test
    res = test(model, test_data_loader, scaler=scaler)
    # serialize test data
    np.savez_compressed('data/results/dcrnn_predictions.npz', **res)
    print('Predictions saved as {}.'.format('saved/results/dcrnn_predictions.npz'))

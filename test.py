import argparse
import torch
import numpy as np
from lib import metrics
import model.dcrnn_model as module_arch
from parse_config import ConfigParser
from lib import utils
from tqdm import tqdm
import math
import time


def main(config):
    logger = config.get_logger('test')

    graph_pkl_filename = 'data/sensor_graph/adj_mx_unix.pkl'
    _, _, adj_mat = utils.load_graph_data(graph_pkl_filename)
    data = utils.load_dataset(dataset_dir='data/METR-LA',
                              batch_size=config["arch"]["args"]["batch_size"],
                              test_batch_size=config["arch"]["args"]["batch_size"])
    test_data_loader = data['test_loader']
    scaler = data['scaler']
    num_test_iteration= math.ceil(data['x_test'].shape[0] / config["arch"]["args"]["batch_size"])

    # build model architecture
    adj_arg = {"adj_mat": adj_mat}
    model = config.initialize('arch', module_arch, **adj_arg)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    y_preds = torch.FloatTensor([])
    y_truths = data['y_test']  # (6850, 12, 207, 2)
    y_truths = scaler.inverse_transform(y_truths)
    predictions = []
    groundtruth = list()

    start_time = time.time()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_data_loader.get_iterator()), total=num_test_iteration):
            x = torch.FloatTensor(x).cuda()
            y = torch.FloatTensor(y).cuda()
            outputs = model(x, y, 0)  # (seq_length, batch_size, num_nodes*output_dim)  (12, 50, 207*1)
            y_preds = torch.cat([y_preds, outputs], dim=1)
    inference_time = time.time() - start_time
    logger.info("Inference time: {:.4f} s".format(inference_time))
    y_preds = torch.transpose(y_preds, 0, 1)
    y_preds = y_preds.detach().numpy()  # cast to numpy array
    print("--------test results--------")
    for horizon_i in range(y_truths.shape[1]):
        y_truth = np.squeeze(y_truths[:, horizon_i, :, 0])

        y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :])
        predictions.append(y_pred)
        groundtruth.append(y_truth)

        mae = metrics.masked_mae_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        mape = metrics.masked_mape_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        rmse = metrics.masked_rmse_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        print(
            "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                horizon_i + 1, mae, mape, rmse
            )
        )
        log = {"Horizon": horizon_i+1, "MAE": mae, "MAPE": mape, "RMSE": rmse}
        logger.info(log)
    outputs = {
        'predictions': predictions,
        'groundtruth': groundtruth
    }

    # serialize test data
    np.savez_compressed('saved/results/dcrnn_predictions.npz', **outputs)
    print('Predictions saved as {}.'.format('saved/results/dcrnn_predictions.npz'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch DCRNN')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)

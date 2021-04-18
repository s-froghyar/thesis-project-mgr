from config import SegmentedCNNconfig, TpCNNconfig, AugerinoCNNconfig
from tpreporter import Reporter
from model_fitter import ModelFitter
import argparse
import torch
from datetime import date

model_configs = {
    'segmented': SegmentedCNNconfig,
    'tp': TpCNNconfig,
    'augerino': AugerinoCNNconfig,
}

parser = argparse.ArgumentParser(description='Run training and evaluation on augmentation techniques in GTZAN dataset')

parser.add_argument('--config', '-c', required=True, choices=list(model_configs.keys()), help='The config string to use')
parser.add_argument('--data-path', '-d', required=True, help='The path to the dataset')
parser.add_argument('--transform', '-t', required=True, choices=['ni', 'ps', 'none'], help='The audio transformation to use')
parser.add_argument('--epoch', '-e', required=True, choices=[str(x) for x in range(100)], help='The epoch number to train')
parser.add_argument('--local', '-l', action='store_true', help='Don\'t use GPU and use carriage return for logging')
parser.add_argument('--checkpoint', default='checkpoints', help='the path to save checkpointed models to')
parser.add_argument('--title', default='Straight learning', help='Enable crossvalidation')
parser.add_argument('--test-only', default=False, action='store_true', help='Load the saved models from a training round and perform evaluation')

def main():
    args = parser.parse_args()
    config_name = args.config
    if config_name is None:
        raise ValueError('No config name passed to script!')
    print('Grabbing model_config')
    config = model_configs[config_name]
    config.local = args.local
    config.epochs = int(args.epoch)
    if args.local:
        config.epochs = 4
        config.batch_size = 4

    config.aug_params.set_chosen_transform(args.transform)

    print(f'Using config {config_name}, transformation: {args.transform}, local: ${args.local}')

    use_cuda = torch.cuda.is_available() and not args.local
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device {device}')

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    save_directory = f'{args.checkpoint}/{date.today().strftime("%Y-%m-%d")}'

    print(f'Save directory: {save_directory}')

    reporter = Reporter(config_name, config, save_directory, args.title)

    model_fitter = ModelFitter(args, config, device, kwargs, reporter)

    if not args.test_only:
        model_fitter.fit()




if __name__ == "__main__":    
    main()

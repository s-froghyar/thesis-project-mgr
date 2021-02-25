from config import BaselineCNNconfig, SegmentedCNNconfig, TpCNNconfig, AugerinoCNNconfig
from tpreporter import Reporter
from model_fitter import ModelFitter
import argparse
import torch
from datetime import date

model_configs = {
    'baseline': BaselineCNNconfig,
    'segmented': SegmentedCNNconfig,
    'tp': TpCNNconfig,
    'augerino': AugerinoCNNconfig,
}

parser = argparse.ArgumentParser(description='Run training and evaluation on augmentation techniques in GTZAN dataset')

parser.add_argument('--config', '-c', required=True, choices=list(model_configs.keys()), help='The config string to use')
parser.add_argument('--data-path', '-d', required=True, help='The path to the dataset')
parser.add_argument('--local', '-l', action='store_true', help='Don\'t use GPU and use carriage return for logging')
parser.add_argument('--checkpoint', default='checkpoints', help='the path to save checkpointed models to')
parser.add_argument('--crossval', action='store_true', help='Enable crossvalidation')
parser.add_argument('--test-only', default=False, action='store_true', help='Load the saved models from a training round and perform evaluation')

def main():
    args = parser.parse_args()
    config_name = args.config
    if config_name is None:
        raise ValueError('No config name passed to script!')
    print('Grabbing model_config')
    config = model_configs[config_name]
    print(f'Using config {config_name}, local: ${args.local}')

    use_cuda = torch.cuda.is_available() and not args.local
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device {device}')

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    save_directory = f'{args.checkpoint}/{date.today().strftime("%d-%m-%Y")}'

    print(f'Save directory: {save_directory}')

    reporter = Reporter(config_name, config.epochs, save_directory)

    model_fitter = ModelFitter(args, config, device, kwargs, reporter)

    if not args.test_only:
        model_fitter.fit()

    train_num_correct, test_num_correct = model_fitter.evaluate()
    print(train_num_correct, test_num_correct)
    # reporter.keep_log(f'FINAL AVERAGE LOSS: {sum(losses) / len(losses)}')







if __name__ == "__main__":    
    main()

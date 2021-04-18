import argparse
import torch
from model_fitter import Interpreter
from config import *


model_configs = {
    'no_aug': SegmentedCNNconfig,
    'segmented': SegmentedCNNconfig,
    'tp': TpCNNconfig,
    'augerino': AugerinoCNNconfig,
}

parser = argparse.ArgumentParser(description='Run evaluation on augmentation techniques in GTZAN dataset')

parser.add_argument('--config', '-c', required=True, choices=['no_aug', 'segmented', 'tp', 'augerino'], help='The config string to use')
parser.add_argument('--transform', '-t', required=True, choices=['ni', 'ps'], help='The audio transformation to use')
parser.add_argument('--run', '-r', required=True, choices=['1', '2', '3'], help='The run id to use')


def main():
    args = parser.parse_args()

    data_path = 'data'
    model_path = f"data_interpreter/{args.config}"
    config = model_configs[args.config]
    config.aug_params.set_chosen_transform(args.transform)
    if args.config != 'no_aug':
        model_path = f"{model_path}/{args.transform.upper()}"

    model_path = f"{model_path}/run_{args.run}/efinal_model.pt"
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    interpreter = Interpreter((config.model, state_dict), config.aug_params, config.model_type)
    results = interpreter.run_evaluation()

    torch.save(results, f"data_interpreter/{args.config}/run_{args.run}/results.pt")
    print('Done!')
    print(f"No augmentation accuracy: {results['no_aug']['accuracy']}")
    print(f"TTA normal accuracy: {results['tta_normal']['accuracy']}")



if __name__ == "__main__":    
    main()

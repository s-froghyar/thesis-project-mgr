import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from reporter import Reporter
from dataset import load_data, GtzanDataset
from baseline_cnn import CNN

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reporter = Reporter('baseline')
    is_test = True

    # Load Data
    GTZAN = load_data(is_test=is_test)

    train_dataset = GtzanDataset(GTZAN.train_x, GTZAN.train_y, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)

    test_dataset = GtzanDataset(GTZAN.test_x, GTZAN.test_y, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize network
    model = CNN('yeet', get_report_data=True).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Train Network
    for epoch in range(args.epochs):
        print('epoch num ', epoch)
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
    
    reporter.set_post_training_values(model, train_loader)

    # check_accuracy(train_loader, model, device)
    # check_accuracy(test_loader, model, device)

# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model, device):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline CNN")

    parser.add_argument(
        "--dir",
        type=str,
        default='./saved-outputs',
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    args = parser.parse_args()
    main(args)


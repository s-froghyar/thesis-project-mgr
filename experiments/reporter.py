
class Reporter():
    __init__(
        name
    ):
        self.train_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/train')
        self.test_summary_writer = SummaryWriter('logs/tensorboard/' + name + '/test')

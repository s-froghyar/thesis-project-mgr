def report_on_model(self):
    print('\n\Getting report on model...')
    train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64)
    
    self.train_predictions = get_all_preds(self.model, train_loader)
    self.test_predictions = get_all_preds(self.model, test_loader)

    train_num_correct = get_num_correct(self.train_predictions, self.train_set.targets)
    test_num_correct = get_num_correct(self.test_predictions, self.test_set.targets)
    
    self.train_confusion_matrix = confusion_matrix(
                                self.train_set.targets,
                                self.train_predictions.argmax(dim=1))
    self.test_confusion_matrix = confusion_matrix(
                                self.test_set.targets,
                                self.test_predictions.argmax(dim=1))
    
    return (train_num_correct, test_num_correct)
import torch
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def compute_accuracy(model, data_loader, device):
    
    train_accuracy = Accuracy(task='BINARY').to(device)

    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1) # returns (max, max_indices)

            train_accuracy(predicted_labels, targets)

    # total train accuracy over all training batches
    total_train_accuracy = train_accuracy.compute()
    train_accuracy.reset()

    return total_train_accuracy
            

def compute_confusion_matrix(model, data_loader, device):

    all_targets, all_predictions = [], []
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            model = model.to(device)
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            # save all the targets and predictions
            all_targets.extend(targets.to('cpu').numpy())
            all_predictions.extend(predicted_labels.to('cpu').numpy())

    cm = confusion_matrix(all_predictions, all_targets)
    
    return cm

def show_misclassified(model, data_loader, device):

    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            model = model.to(device)
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            unmatched = torch.where(targets != predictions)[0]

            for idx in unmatched:
                plt.figure()
                features_unmatched = features[idx]
                nhwc_img = features_unmatched.to('cpu')
                plt.imshow(nhwc_img[0])
                plt.title(f"Predicted: {predictions[idx]} | Target: {targets[idx]}")
                plt.show()
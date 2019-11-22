import torch


def compute_accuracy(target, output):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Tensor of model predictions.
            It should have the same dimensions as target
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(torch.argmax(target, dim=1) == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy

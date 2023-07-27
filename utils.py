import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def train_one_epoch(model, train_iterator, optim, loss_function, save_params=False, device="cpu"):
    model.train()

    for i, (qa, qid, labels, mask) in enumerate(train_iterator):
        qa, qid, labels, mask = (
            qa.to(device),
            qid.to(device),
            labels.to(device),
            mask.to(device),
        )

        optim.zero_grad()
        pred = model(qid, qa)
        loss = loss_function(pred, labels, mask)
        loss.backward()
        optim.step()
    if(save_params):
        torch.save(model.state_dict(), "model_params.pkl")


def eval_one_epoch(model, test_iterator, device):
    """
    evaluate on test set and print the auc score
    """
    model.eval()

    preds = []
    truths = []
    binary_preds_no_first = []
    preds_no_first = []
    truths_no_first = []

    for i, (qa, qid, labels, mask) in enumerate(test_iterator):
        qa, qid, labels, mask = (
            qa.to(device),
            qid.to(device),
            labels.to(device),
            mask.to(device),
        )

        with torch.no_grad():
            pred = model(qid, qa)

        one_pred, one_truth, truth_no_first, pred_no_first = values_after_mask(pred, labels, mask)
        preds.append(one_pred)
        truths.append(one_truth)
        truths_no_first.append(truth_no_first)
        preds_no_first.append(pred_no_first)

    # on the last pred value
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    # on pred values except the first one
    truths_no_first = np.concatenate(truths_no_first)
    preds_no_first = np.concatenate(preds_no_first)

    auc_on_last = roc_auc_score(truths, preds)
    acc_on_last = accuracy_score(truths, preds.round())
    auc_no_first = roc_auc_score(truths_no_first, preds_no_first)
    acc_no_first = accuracy_score(truths_no_first, preds_no_first.round())

    print("\nauc_on_last=%.4f acc_on_last=%.4f auc_no_first=%.4f acc_no_first=%.4f"%(auc_on_last, acc_on_last, auc_no_first, acc_no_first))



def values_after_mask(pred, labels, mask):
    """
    return what we really need to predict, i.e. the probability of the t-th exercise
    other parts in a sequence are masked
    """
    mask_one_left = mask.gt(0.9)
    mask_one_left = mask_one_left.view(-1)

    mask_except_first = mask.gt(0)
    mask_except_first = mask_except_first.view(-1)
    """
    detach()
    Returns a new Tensor, detached from the current graph.
    """
    last_pred = torch.masked_select(pred.view(-1), mask_one_left).detach().cpu().numpy()
    last_truth = torch.masked_select(labels.view(-1), mask_one_left).detach().cpu().numpy()

    pred_except_first = torch.masked_select(pred.view(-1), mask_except_first).detach().cpu().numpy()
    truth_except_first = torch.masked_select(labels.view(-1), mask_except_first).detach().cpu().numpy()

    return last_pred, last_truth, truth_except_first, pred_except_first
import torch
import pickle
import random
import glob
import numpy as np
from collections import defaultdict


def any2device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    elif isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [any2device(subvalue, device) for subvalue in value]

    assert "Your object type is not implemented"


def save_results(res, filename):
    with open(filename, 'wb') as f:
        pickle.dump(res, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result


def fit_model(model, dataloaders, criterion, optim, args, output_dir):
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = 'cpu'
    best_loss = 100
    criterion.to(device)
    model.to(device)
    for epoch in range(args.num_epochs):
        run_loss = train_epoch(model, dataloaders['train'], criterion, optim, device)
        if run_loss < best_loss:
            best_loss = run_loss
            torch.save(model.to(torch.device("cpu")), output_dir + 'best.pkl')
            model.to(device)

        eval_results = eval_model(model, dataloaders['valid'], criterion, device, 'valid', str(epoch))

    model = torch.load(output_dir + 'best.pkl')
    for name in dataloaders.keys():
        eval_results = eval_model(model, dataloaders[name], criterion, device, name, "after")
        save_results(eval_results, output_dir + name + '_results.pkl')


def train_epoch(model, dataloader, criterion, optim, device):
    model.train()
    run_loss = 0
    for batch in dataloader:
        batch = any2device(batch, device)
        output = model(batch)
        loss = criterion(output, batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        run_loss += loss.item()

    return run_loss / len(dataloader)

def count_metrics(sub_preds, labels):
    acc = np.sum(np.argmax(sub_preds, axis=-1) == labels) / sub_preds.shape[0]
    pressisions = []
    recalls = []
    for lab in set(labels):
        TP = np.sum((np.argmax(sub_preds, axis=-1) == lab) * (labels == lab))
        FP = np.sum((np.argmax(sub_preds, axis=-1) == lab) * (labels != lab))
        FN = np.sum((np.argmax(sub_preds, axis=-1) != lab) * (labels == lab))
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        pressisions.append(precision)
        recalls.append(recall)
    return {
        'acc': acc,
        'precision': np.mean(pressisions),
        'recall': np.mean(recalls)
    }
    
def eval_model(model, dataloader, criterion, device, name, epoch):
    model.eval()
    run_loss = 0
    sub_preds = []
    sub_labels = []
    for batch in dataloader:
        batch = any2device(batch, device)
        with torch.no_grad():
            output = model(batch)
            loss = criterion(output, batch)
            run_loss += loss.item()
            sub_preds.append(output['log_preds'].detach().cpu())
            sub_labels.append(batch['labels'].numpy().reshape(-1,1))
    sub_preds = np.vstack(sub_preds)
    sub_labels = np.vstack(sub_labels).reshape((-1))
    metrics = count_metrics(sub_preds, sub_labels)
    acc = metrics['acc']
    precision = metrics['precision']
    recall = metrics['recall']

    ans = {'loss': run_loss / len(dataloader), **metrics}
    print(epoch, " ", name, "  eval loss  ", ans['loss'], "  eval acc  ", ans['acc'],
        " eval precision  ", ans['precision'], "  eval recall  ", ans['recall']
    )
    
    return ans


def infer_model(model, dataloader, device, key):
    model.eval()
    ans = []
    for batch in dataloader:
        batch = any2device(batch, device)
        with torch.no_grad():
            output = model(batch)
            ans.append(output[key].detach().cpu())
    ans = np.vstack(ans)

    return ans

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def test_model(name, path):

    paths = glob.glob(path + f'exp_{name}_*/test_results.pkl')
    result = defaultdict(list)
    
    for path in paths:
        res = load_results(path)
        for k in res:
            result[k].append(res[k])    
    return result

def print_result(name, results):

    print(name)
    for k in results:
        if k == 'loss':
            continue
        print(f'\t {k}: {round(np.mean(results[k]), 3)} +- {round(np.std(results[k]), 3)}')
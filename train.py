from tqdm import tqdm
import torch
import numpy as np
from sklearn import metrics

def train_eval(cate,model, data_loader,optimizer,loss_func):
    model.train() if cate == 'train' else model.eval()
    acc, loss_sum = 0.0, 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for d in tqdm(data_loader,desc=cate):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_func(outputs, targets)

        if cate == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc += outputs.max(dim=1)[1].eq(targets).sum().data
        loss_sum += loss.data

        labels = targets.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    acc = acc * 100 / len(data_loader.dataset)
    loss_sum = loss_sum / len(data_loader)
    print(metrics.classification_report(labels_all,predict_all))
    return acc, loss_sum
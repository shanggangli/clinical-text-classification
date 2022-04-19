import torch
from model import bert,bert_cnn,bert_rnn
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import train
from pytorch_pretrained_bert import optimization
import torch.nn.functional as F

if __name__ == '__main__':
    # hyper-parameter
    BATCH_SIZE = 8
    RANDOM_SEED = 1000
    MAX_LEN = 512
    learning_rate = 0.001
    epochs = 10
    min_loss = 1000
    path = "/Users/lee/Desktop/text classification/cleaned_text0320.csv"
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    data = pd.read_csv(path)
    data.dropna(axis=0, how="any", inplace=True)
    data = data[0:100]

    df_train, df_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

    bert_save_path = "bert_best_model_LCX.bin"
    train_data_loader = utils.create_data_loader(df_train, tokenizer, MAX_LEN,"LCX", BATCH_SIZE)
    test_data_loader = utils.create_data_loader(df_test, tokenizer, MAX_LEN,"LCX", BATCH_SIZE)

    # model
    model = bert.bertModel(n_classes = 2)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss function
    loss_F = F.cross_entropy

    for i in range(0,epochs):
        print("----------- epochs:{0}/{1} -----------".format(i,epochs))
        # training
        print("----------- train -----------")
        train_acc,train_loss = train.train_eval("train",model,train_data_loader,optimizer,loss_F)
        print("train acc:{0}   train loss:{1}".format(train_acc,train_loss))
        # test
        print("----------- test -----------")
        test_acc, test_loss = train.train_eval("test", model, test_data_loader, optimizer, loss_F)
        print("test acc:{0}   test loss:{1}".format(train_acc,train_loss))

        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), bert_save_path)
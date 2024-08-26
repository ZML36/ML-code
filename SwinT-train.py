from datasets import ClassificationDataset
from metrics import ClassificationMetrics
from torch import optim
import torch
import numpy as np
import os
import time
from network.Vision_Transformers import creat_transformers
import torch.nn as nn
from torch.nn import functional as F


def train(model, device, criterion, optimizer, train_loader):
    # 训练开始时间计算
    start = time.time()
    train_pred = []
    train_prob = []
    train_label = []
    train_loss = []
    model.train()

    for item, (image, label) in enumerate(train_loader):
        images,labels = image.float().to(device), label.squeeze(1).to(device)
        outputs = model(images)  

        probs = torch.sigmoid(outputs)  # 将模型输出转换为概率值
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, dim=1)
        train_pred.extend(predicted.data.cpu().numpy())
        train_label.extend(labels.data.cpu().numpy())
        train_prob.extend(probs.data.cpu().numpy())  # 保存正类的概率值
        train_loss.append(loss.item())

    end = time.time()
    print("Run epoch time: %.2fs"%(end - start))
    return train_pred, train_label, train_prob, train_loss

def val(model, device, criterion, test_loader):
    model.eval()
    test_pred = []
    test_label = []
    test_loss = []
    test_prob = []
    with torch.no_grad():
        for item,(image, label) in enumerate(test_loader):
            images = image.float().to(device)
            labels = label.squeeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # 将模型输出转换为概率值
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, dim=1)
            test_pred.extend(predicted.data.cpu().numpy())
            test_label.extend(labels.data.cpu().numpy())
            test_prob.extend(probs.data.cpu().numpy())  # 保存正类的概率值
            test_loss.append(loss.item())

    return test_pred, test_label, test_prob, test_loss


def train_net(model, device, train_path, test_path, image_size=512, epochs=40, train_batch=4, test_batch=8, lr=0.001, save_path=None, resume=None):
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])
        print(f"loading model from {resume}, epoch: {checkpoint['epoch']}")
        star_epoch = checkpoint['epoch']
    else:
        star_epoch = 0

    model.to(device)
    # 加载训练集
    train_dataset = ClassificationDataset(train_path, image_size=image_size, aug=True, requires_name=False)
    test_dataset = ClassificationDataset(test_path, image_size=image_size, aug=False, requires_name=False)
    print("train data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
 
    print('*'*30,'开始训练','*'*30)
    best_acc = 0
    for epoch in range(star_epoch, epochs):
        #训练
        train_pred, train_label, train_prob, train_loss = train(model, device, criterion, optimizer, train_loader)
        #验证
        test_pred, test_label, test_prob, test_loss = val(model, device, criterion, test_loader)
        
        print('Epoch:%s/%s'%(epoch + 1, epochs),'丨Train loss: %.4f丨'%np.mean(train_loss), '丨Test loss: %.4f丨'%np.mean(test_loss))

        #train
        metrics = ClassificationMetrics(train_label, train_pred, train_prob, 3)
        acc, recall, prec, auc = metrics.accuracy(), metrics.recall(),metrics.precision(),metrics.auc()
        print('%s:丨acc: %.4f丨丨recall: %.4f丨丨prec: %.4f丨丨auc: label0: %.4f | label1: %.4f | label2: %.4f丨' %("Train", acc, recall, prec, auc[0], auc[1], auc[2]))

        #test
        metrics = ClassificationMetrics(test_label, test_pred, test_prob, 3)
        acc, recall, prec, auc = metrics.accuracy(), metrics.recall(),metrics.precision(),metrics.auc()
        print('%s:丨acc: %.4f丨丨recall: %.4f丨丨prec: %.4f丨丨auc: label0: %.4f | label1: %.4f | label2: %.4f丨' %("Test", acc, recall, prec, auc[0], auc[1], auc[2]))

        if acc > best_acc:
            best_acc = acc
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            print("saving epoch {} model to {} ".format(epoch, save_path))
            torch.save(state, save_path)

        print("="*100)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    image_size = 224
    model_name = "swint"  
    model = creat_transformers(model_name, num_classes=3, pretrained=False)
    train_path ="dataset/train/"
    test_path = "dataset/test/"
    pth_path = f'{model_name}.pth'
    train_net(model, 
              device, 
              train_path, 
              test_path, 
              image_size=image_size, 
              epochs=200,
              train_batch=32, 
              test_batch=32, 
              lr=0.0001, 
              save_path=pth_path, 
              resume=None)
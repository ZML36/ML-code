from datasets import ClassificationDataset,ClassificationDataset_without_label
from metrics import ClassificationMetrics
import torch
from network.Vision_Transformers import creat_transformers
from torch.nn import functional as F
import logging
from datetime import datetime
import pandas as pd
import torch.nn as nn
def model_pred_withlabel(model, device, test_loader, save_path, num_classes):
    logging.basicConfig(filename=save_path, level=logging.INFO)
    model.to(device)
    model.eval()
    test_pred = []
    test_label = []
    test_prob = []
    df = pd.DataFrame()
    with torch.no_grad():
        for item,(image, label, name) in enumerate(test_loader):
            images = image.float().to(device)
            labels = label.squeeze(1).to(device)
            outputs = model(images)

            tensor_before_classification = model.forward_features(images).squeeze(0).cpu().numpy()
            print(tensor_before_classification.shape)
            row_data = [name[0]] + tensor_before_classification.tolist()
            df = df.append(pd.Series(row_data), ignore_index=True)

            probs = torch.sigmoid(outputs)  # 将模型输出转换为概率值
            _, predicted = torch.max(outputs, dim=1)

            test_pred.extend(predicted.data.cpu().numpy())
            test_label.extend(labels.data.cpu().numpy())
            test_prob.extend(outputs.data.cpu().numpy())  # 保存正类的概率值

            logging.info(f"image name: {name[0]}, pred_class: {int(torch.argmax(outputs, 1).data.cpu().numpy())}, 'pred_prob': {float(probs[:, int(torch.argmax(outputs, 1).data.cpu().numpy())].data.cpu().numpy())}")

    

    num_columns = len(tensor_before_classification)
    column_names = ['Name'] + [f'Column_{i+1}' for i in range(num_columns)]
    df.columns = column_names
    df.to_excel(save_path.replace('.log','.xlsx').replace('test_file','before_classification'), index=False)
    
    metrics = ClassificationMetrics(test_label, test_pred, test_prob, num_classes)
    acc, recall, prec, auc = metrics.accuracy(), metrics.recall(),metrics.precision(),metrics.auc()
    print('%s:丨acc: %.4f丨丨recall: %.4f丨丨prec: %.4f丨丨auc: %s丨' %("Test", acc, recall, prec, auc))
    print('结束预测')

def model_pred_withoutlabel(model, device, test_loader, save_path):
    logging.basicConfig(filename=save_path, level=logging.INFO)
    model.to(device)
    model.eval()
    test_pred = []
    test_prob = []
    pred_dict = {}
    df = pd.DataFrame()
    with torch.no_grad():
        for item, (image, name) in enumerate(test_loader):
            images = image.float().to(device)
            outputs = model(images)

            tensor_before_classification = model.forward_features(images).squeeze(0).cpu().numpy()
            print(tensor_before_classification.shape)
            row_data = [name[0]] + tensor_before_classification.tolist()
            df = df.append(pd.Series(row_data), ignore_index=True)


            probs = torch.sigmoid(outputs)  # 将模型输出转换为概率值
            _, predicted = torch.max(outputs, dim=1)
            test_pred.extend(predicted.data.cpu().numpy())
            test_prob.extend(probs[:, 1].data.cpu().numpy())  # 保存正类的概率值
            pred_dict[name[0]] = {'pred_class':int(torch.argmax(outputs, 1).data.cpu().numpy()), 'pred_prob': float(probs[:, int(torch.argmax(outputs, 1).data.cpu().numpy())].data.cpu().numpy())}
            logging.info(f"image name: {name[0]}, pred_class: {int(torch.argmax(outputs, 1).data.cpu().numpy())}, 'pred_prob': {float(probs[:, int(torch.argmax(outputs, 1).data.cpu().numpy())].data.cpu().numpy())}")


    num_columns = len(tensor_before_classification)
    column_names = ['Name'] + [f'Column_{i+1}' for i in range(num_columns)]
    df.columns = column_names
    df.to_excel(save_path.replace('.log','.xlsx').replace('test_file','before_classification'), index=False)
    print('结束预测')
   

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    model_name = "swint"  
    num_classes = 3
    model = creat_transformers(model_name, num_classes=num_classes, pretrained=False)
    test_path = r"dataset/ANYI"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{model_name}_test_file-{current_time}.log"
    pertrained_path = f'{model_name}.pth'
    model.load_state_dict(torch.load(pertrained_path)['model'])
    
    test_dataset = ClassificationDataset_without_label(test_path, image_size=image_size, aug=False, requires_name=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model_pred_withoutlabel(model, device, test_loader, save_path=save_path)
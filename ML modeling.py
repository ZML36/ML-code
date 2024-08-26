import pandas as pd
from sklearn import preprocessing
import tkinter as Tk
from tkinter import filedialog
from tkinter.simpledialog import askinteger
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn import preprocessing
from tkinter import Tk, filedialog
from tkinter.simpledialog import askinteger
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
root = Tk()
root.withdraw()  # 隐藏主窗口
group_names = ['0', '1', '2']



# 弹出对话框让用户选择训练集文件
train_filename = filedialog.askopenfilename(title="请选择训练集文件", filetypes=[("Excel files", "*.xlsx;*.xls")])

# 读取excel文件
df_train = pd.read_excel(train_filename)

# 选择所有的列作为特征进行建模
X_train = df_train.copy()  # 创建df_train的副本

# 对第三列及以后的列进行正则化
for col in X_train.columns[2:]:
    X_train[col] = preprocessing.scale(X_train[col])

# 输出结果
print(X_train)

# 弹出对话框让用户输入验证集的数量
num_validation_sets = askinteger("输入", "请输入验证集的数量：", minvalue=1)

# 循环读取用户指定数量的验证集文件
validation_sets = []
for i in range(num_validation_sets):
    validation_filename = filedialog.askopenfilename(title=f"请选择验证集{i+1}文件", filetypes=[("Excel files", "*.xlsx;*.xls")])
    
    # 读取excel文件
    df_validation = pd.read_excel(validation_filename)
    
    # 根据训练集的列名选择对应的列
    selected_columns = df_train.columns.intersection(df_validation.columns)
    
    # 创建验证集DataFrame
    df_validation_selected = df_validation[selected_columns]

    # 对第三列及以后的列进行正则化（和训练集相同的处理）
    for col in df_validation_selected.columns[2:]:
        df_validation_selected[col] = preprocessing.scale(df_validation_selected[col].astype(float))
    
    # 将处理后的验证集DataFrame添加到列表中
    validation_sets.append(df_validation_selected)

# 打印验证集结果
for i, df in enumerate(validation_sets, start=1):
    print(f"验证集{i}:")
    print(df)
import pandas as pd
from sklearn.model_selection import train_test_split

# 假设您已经有了一个DataFrame df_train，其中包含训练数据，以及一个validation_sets列表，其中包含所有的验证集DataFrame

# 定义训练集
X_train = df_train.iloc[:, 2:]  # 选择第三列及以后的列

y_train = df_train['Group']

# 初始化字典来存储验证集的特征矩阵和目标向量
# 初始化字典来存储验证集的特征矩阵和目标向量
validation_data = {}

# 遍历所有验证集DataFrame
for i, df_validation in enumerate(validation_sets, start=1):
    # 定义验证集特征矩阵和目标向量
    X_test = df_validation.drop(columns=['Group', 'imageName'])
    y_test = df_validation['Group']
    
    # 将验证集数据存储在字典中
    validation_data[f'X_test{i}'] = X_test
    validation_data[f'y_test{i}'] = y_test


# 现在，validation_data字典包含了所有的验证集特征矩阵和目标向量，可以通过键名访问

# 例如，访问第一个验证集的特征矩阵和目标向量
X_test1 = validation_data['X_test1']
y_test1 = validation_data['y_test1']

# 如果需要，可以循环遍历字典来访问所有的验证集数据
for i in range(1, num_validation_sets + 1):
    X_test = validation_data[f'X_test{i}']
    y_test = validation_data[f'y_test{i}']
    # 在这里，您可以使用X_test和y_test进行模型评估或其他操作
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV
#SVM
svm =SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, 
cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svm.fit(X_train, y_train)

# 模型定义和参数搜索
# SVM - 直接定义，不进行参数搜索
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, 
          probability=True, tol=0.001, cache_size=200, class_weight=None, 
          verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svm.fit(X_train, y_train)

# SGD
sgd = SGDClassifier()
param_grid_sgd = {
    'loss': ['log_loss'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.001, 0.01, 0.1, 1]
}
sgd_search = GridSearchCV(sgd, param_grid_sgd, cv=5)
sgd_search.fit(X_train, y_train)
sgd = GridSearchCV(sgd, param_grid_sgd, cv=5)
sgd.fit(X_train, y_train)
print("Best parameters:", sgd.best_params_)

# KNN
knn = KNeighborsClassifier()
param_dist_knn = {
    'n_neighbors': scipy.stats.randint(1, 10),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree', 'brute']
}
knn_search = RandomizedSearchCV(knn, param_distributions=param_dist_knn, cv=5, n_iter=10)
knn_search.fit(X_train, y_train)
knn = RandomizedSearchCV(knn, param_distributions=param_dist_knn, cv=5)
knn.fit(X_train, y_train)
print("Best parameters:", knn.best_params_)

# 随机森林
rf = RandomForestClassifier()
param_dist_rf = {
    'n_estimators': scipy.stats.randint(1, 50),
    'max_depth': scipy.stats.randint(1, 10),
    'min_samples_split': scipy.stats.randint(2, 10)
}
rf_search = RandomizedSearchCV(rf, param_distributions=param_dist_rf, cv=5, n_iter=10)
rf_search.fit(X_train, y_train)
rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, cv=5)
rf.fit(X_train, y_train)
print("Best parameters:", rf.best_params_)
rf = rf.best_estimator_

# XGBoost
xgb = XGBClassifier()
param_dist_xgb = {
    'n_estimators': scipy.stats.randint(1, 50),
    'max_depth': scipy.stats.randint(1, 10),
    'learning_rate': scipy.stats.uniform(0.01, 0.5)
}
xgb_search = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, cv=5, n_iter=10)
xgb_search.fit(X_train, y_train)
xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, cv=5)
xgb.fit(X_train, y_train)
print("Best parameters:", xgb.best_params_)
xgb = xgb.best_estimator_

# LightGBM
lgb = LGBMClassifier()
param_dist_lgb = {
    'n_estimators': scipy.stats.randint(1, 50),
    'max_depth': scipy.stats.randint(1, 10),
    'learning_rate': scipy.stats.uniform(0.01, 0.5)
}
lgb_search = RandomizedSearchCV(lgb, param_distributions=param_dist_lgb, cv=5, n_iter=10)
lgb_search.fit(X_train, y_train)
lgb = RandomizedSearchCV(lgb, param_distributions=param_dist_lgb, cv=5)
lgb.fit(X_train, y_train)
print("Best parameters:", lgb.best_params_)
lgb = lgb.best_estimator_

# 初始化列表来存储最佳参数
best_params_list = []

# 添加每个模型的最佳参数
model_names = ['SVM', 'SGD', 'KNN', 'RandomForest', 'XGBoost', 'LightGBM']
models_search = [svm, sgd_search, knn_search, rf_search, xgb_search, lgb_search]

# 手动添加SVM的参数，因为它没有进行搜索
best_params_list.append(['SVM', {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'}])

for name, model in zip(model_names[1:], models_search[1:]):  # 从第二个开始，因为第一个是SVM
    best_params_list.append([name, model.best_params_])

# 创建一个DataFrame
best_params_df = pd.DataFrame(best_params_list, columns=['Model', 'Best Parameters'])

# 将DataFrame转换为Excel文件
best_params_df.to_excel('best_model_parameters.xlsx', index=False)


model_names = ['SVM', 'SGD', 'KNN', 'RandomForest', 'XGBoost', 'LightGBM']
models = [svm, sgd, knn, rf, xgb, lgb]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# 初始化列表来存储结果
result_table = []

# 对于每个模型，计算在训练集和每个验证集上的性能指标
for name, model in zip(model_names, models):
    # 训练集
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_f1 = f1_score(y_train, y_train_pred, average='macro')

    # 验证集
    for j in range(1, num_validation_sets + 1):
        X_test = validation_data[f'X_test{j}']
        y_test = validation_data[f'y_test{j}']
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='macro')
        test_recall = recall_score(y_test, y_test_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        # 将结果添加到列表中
        result_table.append([name, j, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1])

# 将结果转化成DataFrame并导出为Excel文件
result_df = pd.DataFrame(result_table, columns=['Model', 'Validation Set', 'Train Accuracy', 'Test Accuracy', 'Train Precision', 'Test Precision', 'Train Recall', 'Test Recall', 'Train F1-score', 'Test F1-score'])
result_df.to_excel('model_performance.xlsx', index=False)
result_df


# 定义函数绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, group_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    thresh = cm.max() / 2.
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            plt.text(k, j, cm[j, k], ha="center", va="center", color="white" if cm[j, k] > thresh else "black")
    plt.xticks(np.arange(len(group_names)), group_names, rotation=45)
    plt.yticks(np.arange(len(group_names)), group_names)
    plt.xlabel('Predicted group')
    plt.ylabel('True group')
    plt.title(f'Confusion Matrix for {model_name} on {dataset_name}')
    plt.savefig(f'./{dataset_name}-混淆矩阵-{model_name}.svg', format='svg', dpi=1200, bbox_inches='tight')
    plt.savefig(f'./{dataset_name}-混淆矩阵-{model_name}.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    #plt.show()

# 对于训练集和每个验证集，绘制每个模型的混淆矩阵
for i, model in enumerate(models):
    # 训练集
    y_train_pred = model.predict(X_train)
    plot_confusion_matrix(y_train, y_train_pred, model_names[i], 'Train', group_names)

    # 验证集
    for j in range(1, num_validation_sets + 1):
        X_test = validation_data[f'X_test{j}']
        y_test = validation_data[f'y_test{j}']
        y_test_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_test_pred, model_names[i], f'Validation{j}', group_names)




import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# 定义函数，用于画多分类的ROC曲线和计算AUC
def plot_roc_curve(model, model_name, X, y, dataset_name, n_classes=3):
    # 获取模型的预测概率值
    probas_ = model.predict_proba(X)

    # 将目标变量二值化（将三类转成三列）
    y_bin = label_binarize(y, classes=[0, 1, 2])

    # 计算每个类别和微、宏平均的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probas_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probas_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘制ROC曲线和标记AUC值
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {:.3f}, 95% CI: {:.3f}-{:.3f})'.format(
                 roc_auc["micro"], roc_auc["micro"] - 1.96 * (roc_auc["micro"] * (1 - roc_auc["micro"]) / len(y)) ** 0.5,
                 roc_auc["micro"] + 1.96 * (roc_auc["micro"] * (1 - roc_auc["micro"]) / len(y)) ** 0.5),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (AUC = {:.3f}, 95% CI: {:.3f}-{:.3f})'.format(
                 roc_auc["macro"], roc_auc["macro"] - 1.96 * (roc_auc["macro"] * (1 - roc_auc["macro"]) / len(y)) ** 0.5,
                 roc_auc["macro"] + 1.96 * (roc_auc["macro"] * (1 - roc_auc["macro"]) / len(y)) ** 0.5),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {} (AUC = {:.3f}, 95% CI: {:.3f}-{:.3f})'.format(
                     i, roc_auc[i], roc_auc[i] - 1.96 * (roc_auc[i] * (1 - roc_auc[i]) / len(y)) ** 0.5,
                     roc_auc[i] + 1.96 * (roc_auc[i] * (1 - roc_auc[i]) / len(y)) ** 0.5))

    # 设置图例、坐标轴、标题
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(model_name))

    # 保存图片
    path = f"./roc_curve/{dataset_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, f"{model_name}-ROC.pdf")
    plt.savefig(full_path, format='pdf', dpi=1200, bbox_inches='tight')
    print(f"ROC曲线已保存至{full_path}。")
    plt.close()  # 关闭图像，避免在Jupyter中显示

    # 返回预测概率值用于保存到Excel
    return probas_

# 准备保存到Excel的数据
results = []

# 对于训练集和每个验证集，绘制每个模型的ROC曲线
for i, model in enumerate(models):
# 训练集
    probas_train = plot_roc_curve(model, model_names[i], X_train, y_train, 'Train')
    results.append(pd.DataFrame(probas_train, columns=[f"{model_names[i]}_train_class{j}" for j in range(3)]))

    # 验证集
    for j in range(1, num_validation_sets + 1):
        X_test = validation_data[f'X_test{j}']
        y_test = validation_data[f'y_test{j}']
        probas_test = plot_roc_curve(model, model_names[i], X_test, y_test, f'Validation{j}')
        results.append(pd.DataFrame(probas_test, columns=[f"{model_names[i]}_validation{j}_class{k}" for k in range(3)]))

# 合并所有DataFrame并保存到Excel
final_df = pd.concat(results, axis=1)
final_df.to_excel("model_predictions.xlsx", index=False)


import os
import shap
import matplotlib.pyplot as plt

import shap

# 模型解释器
def explain_model_with_shap(model_names, models, X_tests, X_trains):
    shap.initjs()

    for i, (model_name, model) in enumerate(zip(model_names, models)):
        print(f"Processing SHAP for Model {i + 1} with model {model_name}")

        # 创建SHAP解释器
        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
        elif model_name in ["SVM", "KNN"]:
            # 对于不支持predict_proba的模型，使用predict
            explainer = shap.KernelExplainer(model.predict, X_trains[i])
        else:
            # 对于支持predict_proba的其他模型
            explainer = shap.KernelExplainer(model.predict, X_trains[i])

        shap_values = explainer.shap_values(X_tests[i])

        # 根据SHAP值的结构和模型类型选择正确的SHAP值
        if isinstance(shap_values, list):
            # 多类分类，shap_values是一个列表
            class_index = 2  # 选择要显示的类别索引
            values = shap_values[class_index]
        else:
            # 二元分类或回归，shap_values是一个数组
            values = shap_values

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(values, X_tests[i], show=False)
        plt.savefig(f'./SHAP-Summary-Model{i + 1}-{model_name}.svg', format='svg', dpi=1200, bbox_inches='tight')
        # plt.show()

        # Dependency plot for each feature
        for feature in X_trains[i].columns:
            plt.figure()
            shap.dependence_plot(feature, values, X_tests[i], show=False)
            plt.savefig(f'./SHAP-Dependency-{feature}-Model{i + 1}-{model_name}.svg', format='svg', dpi=1200, bbox_inches='tight')
            # plt.show()

        # SHAP Decision Plots for 10 random samples
        sample_indices = list(range(19, 32))  # 从第20到第30个样本，可以自己定
        for sample in sample_indices:
            plt.figure()
            shap.decision_plot(np.mean(explainer.expected_value), values[sample, :], X_tests[i].columns, link='logit', show=False)
            plt.savefig(f'./SHAP-Decision-Sample{sample}-Model{i + 1}-{model_name}.svg', format='svg', dpi=1200, bbox_inches='tight')
            # plt.show()

# 假定X_train和X_test是已经准备好的数据集
X_trains = [X_train] * len(models)  # 每个模型的训练集
X_tests = [X_test] * len(models)  # 每个模型的测试集

# 运行所有模型的SHAP解释
explain_model_with_shap(model_names, models, X_tests, X_trains)
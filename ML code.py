# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 10:46:05 2026

@author: DELL
"""

#rf
class_weights={
    'clear_air':1,
    'low_cloud':1,
    'mid_high_cloud':1,
    'single_layer_fog':1
}
model = RandomForestClassifier(random_state=42,class_weight=class_weights) 

grid_parameters = {
    'n_estimators':  np.arange(150,800,20), # #树的数量
    'max_features': ['sqrt']     #决策树节点分枝时随机选取特征的数量,共20个特征
    # 'min_samples_split':np.arange(1,20,1),
    # 'min_samples_leaf':np.arange(1,20,1),
    # 'max_depth':np.arange(3,20,1),
}

#SVM

lowerbound=  [0.01,0.001]
upperbound = [100,100]

# 调用粒子群优化算法
best_params, best_score = pso(PSO_function, lowerbound, upperbound, swarmsize=60, maxiter=50)

# 使用PSO优化后的参数创建SVM模型

model = SVC(C=best_params[0], gamma=best_params[1], class_weight=class_weights)

# dt
model = DecisionTreeClassifier(random_state=42,class_weight=class_weights) 
#model = RandomForestClassifier(random_state=42,class_weight= 'balanced_subsample') 

grid_parameters = {
    'min_samples_split':np.arange(1,10,1),
    'min_samples_leaf':np.arange(1,10,1),
    'max_depth':np.arange(3,21,1),
}

# 使用 StratifiedKFold 进行分层抽样
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
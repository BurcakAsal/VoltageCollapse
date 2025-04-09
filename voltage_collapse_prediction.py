import xlrd
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import math
import shap
import lime
import skexplain
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor 

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.models import NodeConfig

from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

from pytorch_tabular import TabularModel

import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 4})
plt.rcParams['figure.figsize'] = [8.0, 8.0]
#plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 5
plt.rcParams['axes.labelsize'] = 20


#arc_df = pd.read_csv('organized_dataarc.csv')

arc_df_1 = pd.read_excel('measurements1.xlsx', sheet_name= 'Sheet1') # Collapse 1
arc_df_0 = pd.read_excel('measurements2.xlsx', sheet_name= 'Sheet1', header=None) # Collapse 0

arc_df_1.columns = arc_df_1.columns.to_series().apply(lambda x: x.strip())

arc_df_1 = arc_df_1.dropna()
arc_df_0 = arc_df_0.dropna()

arc_df_1['Collapse'] = 1
arc_df_0[len(arc_df_0.columns)] = 0

arc_df_1 = arc_df_1.iloc[205:]
arc_df_0 = arc_df_0.iloc[205:]


new_columns = dict(zip(list(arc_df_0.columns), list(arc_df_1.columns)))
arc_df_0 = arc_df_0.rename(index=str, columns=new_columns)


frame_list=[arc_df_1, arc_df_0]
frame_mod=[frame_list[i].iloc[0:] for i in range(0,len(frame_list))]
merged_arc_df=pd.concat(frame_mod, ignore_index=True)

#----------------------------------Data Statistics--------------------------------------

merged_arc_df1 = merged_arc_df[['I1','P1','Q1','V1','Collapse']]
merged_arc_df2 = merged_arc_df[['I2','P2','Q2','V2','Collapse']]
merged_arc_df3 = merged_arc_df[['I3','P3','Q3','V3','Collapse']]

#print(merged_arc_df3)
#print(type(merged_arc_df3))

#print(merged_arc_df1.describe())
print(merged_arc_df1.describe().apply(lambda s: s.apply('{0:.5f}'.format)))
#print(merged_arc_df1.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
#print()
#print(merged_arc_df1.groupby('Collapse').describe().unstack(1))
#print(merged_arc_df1[merged_arc_df1['Collapse'] == 0].describe())
#print()
#print(merged_arc_df1[merged_arc_df1['Collapse'] == 1].describe())

#plt.rcParams['figure.dpi'] = 300
#ax=merged_arc_df1['Collapse'].plot.hist(bins=2)
#ax.set_xticks([0, 1])
"""
merged_arc_df1['Collapse'].plot(
    kind='bar',
    color=merged_arc_df1['Collapse'].replace({0:'green', 1:'red'}),
)
"""
"""
ax = merged_arc_df1.T.plot(kind='bar', label='Collapse', colormap='Paired')
ax.set_xlim(0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(range(len(merged_arc_df1)))
"""
#plt.xlabel('Length') 
#plt.ylabel('Frequency') 
#plt.show()

#------------------------------------------------------------
#arc_df1_summary = pd.DataFrame(merged_arc_df1.describe())
#arc_df2_summary = pd.DataFrame(merged_arc_df2.describe())
#arc_df3_summary = pd.DataFrame(merged_arc_df3.describe())

#fig, ax = plt.subplots()
#ax.table(cellText=arc_df1_summary.values, colLabels=arc_df1_summary.columns, loc='center')
#fig.tight_layout()
#plt.show()
#------------------------------------------------------------
"""
plt.rcParams['figure.dpi'] = 300
sns.heatmap(merged_arc_df1.corr(), annot = True)
plt.rcParams['figure.figsize'] = (7,7)
plt.show()
"""
#----------------------------------Data Statistics--------------------------------------

merged_arc_df['I1'] = (merged_arc_df['I1'] - merged_arc_df['I1'].min()) / (merged_arc_df['I1'].max() - merged_arc_df['I1'].min())
merged_arc_df['I2'] = (merged_arc_df['I2'] - merged_arc_df['I2'].min()) / (merged_arc_df['I2'].max() - merged_arc_df['I2'].min())
merged_arc_df['I3'] = (merged_arc_df['I3'] - merged_arc_df['I3'].min()) / (merged_arc_df['I3'].max() - merged_arc_df['I3'].min())

merged_arc_df['P1'] = (merged_arc_df['P1'] - merged_arc_df['P1'].min()) / (merged_arc_df['P1'].max() - merged_arc_df['P1'].min())
merged_arc_df['P2'] = (merged_arc_df['P2'] - merged_arc_df['P2'].min()) / (merged_arc_df['P2'].max() - merged_arc_df['P2'].min())
merged_arc_df['P3'] = (merged_arc_df['P3'] - merged_arc_df['P3'].min()) / (merged_arc_df['P3'].max() - merged_arc_df['P3'].min())

merged_arc_df['Q1'] = (merged_arc_df['Q1'] - merged_arc_df['Q1'].min()) / (merged_arc_df['Q1'].max() - merged_arc_df['Q1'].min())
merged_arc_df['Q2'] = (merged_arc_df['Q2'] - merged_arc_df['Q2'].min()) / (merged_arc_df['Q2'].max() - merged_arc_df['Q2'].min())
merged_arc_df['Q3'] = (merged_arc_df['Q3'] - merged_arc_df['Q3'].min()) / (merged_arc_df['Q3'].max() - merged_arc_df['Q3'].min())

merged_arc_df['V1'] = (merged_arc_df['V1'] - merged_arc_df['V1'].min()) / (merged_arc_df['V1'].max() - merged_arc_df['V1'].min())
merged_arc_df['V2'] = (merged_arc_df['V2'] - merged_arc_df['V2'].min()) / (merged_arc_df['V2'].max() - merged_arc_df['V2'].min())
merged_arc_df['V3'] = (merged_arc_df['V3'] - merged_arc_df['V3'].min()) / (merged_arc_df['V3'].max() - merged_arc_df['V3'].min())

#arc_phase1 = ['I1','P1','Q1','V1','Collapse']
arc_phase1_features = ['I1','P1','Q1','V1']

#arc_phase2 = ['I2','P2','Q2','V2','Collapse']
arc_phase2_features = ['I2','P2','Q2','V2']

#arc_phase3 = ['I3','P3','Q3','V3','Collapse']
arc_phase3_features = ['I3','P3','Q3','V3']

arc_target = ['Collapse']

X1 = merged_arc_df[arc_phase1_features]
y1 = merged_arc_df[arc_target]

X2 = merged_arc_df[arc_phase2_features]
y2 = merged_arc_df[arc_target]

X3 = merged_arc_df[arc_phase3_features]
y3 = merged_arc_df[arc_target]

#merged_arc_df = pd.concat([arc_df_1,arc_df_0], ignore_index=True)
"""
print(arc_df_1)
print(arc_df_1.dtypes)
print(arc_df_0)
print(arc_df_0.dtypes)
print()
"""

"""
print(merged_arc_df)
print(merged_arc_df.dtypes)
print()
print(X1)
print(y1)
print()
print(X2)
print(y2)
print()
print(X3)
print(y3)
"""

"""
arc_df.loc[:, 'I'] *=1000.0

arc_df.replace({'YOK': 0, 'VAR': 1}, inplace=True)

arc_df['U'] = arc_df['U'].str.replace(',', '.').astype(float)

arc_df['U'] = (arc_df['U'] - arc_df['U'].min()) / (arc_df['U'].max() - arc_df['U'].min())
arc_df['I'] = (arc_df['I'] - arc_df['I'].min()) / (arc_df['I'].max() - arc_df['I'].min())   

arc_features = ['U','I']

arc_target = ['Collapse']

#labels = ['YOK', 'VAR']

X = arc_df[arc_features] # Features
y = arc_df[arc_target]  # Target variable
"""
"""
label_encoder = preprocessing.LabelEncoder()
lbl=label_encoder.fit(labels)
print(lbl.classes_)
y=lbl.transform(y)
"""
#print(arc_df.dtypes)
#print("Label encoder classes",label_encoder.classes_)
#print()
#print(X)
#print()
#print(y)


X1_train, X1_valtest, y1_train, y1_valtest = train_test_split(X1, y1, test_size=0.2, shuffle=True, random_state=42)
X1_val, X1_test, y1_val, y1_test = train_test_split(X1_valtest, y1_valtest, test_size=0.5, shuffle=True, random_state=42)

X2_train, X2_valtest, y2_train, y2_valtest = train_test_split(X2, y2, test_size=0.2, shuffle=True, random_state=42)
X2_val, X2_test, y2_val, y2_test = train_test_split(X2_valtest, y2_valtest, test_size=0.5, shuffle=True, random_state=42)

X3_train, X3_valtest, y3_train, y3_valtest = train_test_split(X3, y3, test_size=0.2, shuffle=True, random_state=42)
X3_val, X3_test, y3_val, y3_test = train_test_split(X3_valtest, y3_valtest, test_size=0.5, shuffle=True, random_state=42)

data1_train =  pd.concat([X1_train, y1_train], axis=1)
data1_val =  pd.concat([X1_val, y1_val], axis=1)
data1_test =  pd.concat([X1_test, y1_test], axis=1)

data2_train =  pd.concat([X2_train, y2_train], axis=1)
data2_val =  pd.concat([X2_val, y2_val], axis=1)
data2_test =  pd.concat([X2_test, y2_test], axis=1)

data3_train =  pd.concat([X3_train, y3_train], axis=1)
data3_val =  pd.concat([X3_val, y3_val], axis=1)
data3_test =  pd.concat([X3_test, y3_test], axis=1)

#print(data_test)



"""
print(X_train)
print()
print(y_train)
print()
print(pd.concat([X_train, y_train], axis=1))
"""

"""
print()
print(X_val)
print()
print(y_val)
print()
print()
print(X_test)
print()
print(y_test)
"""

#--------------------------------------------------------------------------------------------------------------

data_config = DataConfig(
    target=[
        arc_target[0]
    ],  # target should always be a list
    continuous_cols=arc_phase1_features,
)
trainer_config = TrainerConfig(
    batch_size=50,
#    min_epochs=50,
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True,
)

optimizer_config = OptimizerConfig()

model_config_tabnet = TabNetModelConfig(
    task="classification",
#    learning_rate = 0.00001
)

model_config_node = NodeConfig(
    task="classification",
)


tabnet_tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config_tabnet,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)

node_tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config_node,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)



#--------------------------------------------------------------------------------------------------------------
#TabNet

#tabnet_tabular_model.fit(train=data1_train, validation=data1_val)
#tabnet_tabular_model.save_model("smodels/tabnet_b50_e50_phase1_normalized", inference_only=True)
#tabnet_tabular_model=TabularModel.load_model("smodels/tabnet_b50_e50_phase1_normalized")

def shap_tabnet_predict(Test):
	global tabnet_tabular_model
	return tabnet_tabular_model.predict(Test)[['prediction']]#.to_numpy()

def lime_tabnet_predict(Test):
	global tabnet_tabular_model
	return tabnet_tabular_model.predict(pd.DataFrame(Test, columns=['I1','P1','Q1','V1']))[['0_probability','1_probability']].to_numpy()


#print(tabnet_tabular_model)

#pred_df_tabnet = tabnet_tabular_model.predict(data1_test, progress_bar="rich") #X1_test.iloc[[3]]
#print(pred_df_tabnet)
#eval_result = tabnet_tabular_model.evaluate(data1_test)
#tabnet_pred = pred_df_tabnet['prediction']
#print(tabnet_pred.to_numpy())
#print(y1_test.to_numpy().flatten())
#print(classification_report(y1_test.to_numpy().flatten(), tabnet_pred.to_numpy()))
"""
print(X1_test)
print(X1_test.shape)
print()
print(type(pred_df_tabnet))
print(pred_df_tabnet.to_numpy())
print()
print(type(X1_test.iloc[3, :]))
"""
#print(X1_test.shape)
#print(X1_test.columns)
#print(X1_test.iloc[[3]].columns)
#print(X1_test.values[3])

#plt.rcParams['figure.dpi'] = 300
#SHAP
#ex_s = shap.KernelExplainer(shap_tabnet_predict, X1_test[0:100], keep_index=True)
#shap_values = ex_s.shap_values(X1_test[0:100])
#print("Shap Values Shape: ", shap_values.shape)	
#print("Shap Values Shape: ", shap_values[:,:,2].shape)
#print("Expected Value: ", ex_s.expected_value)
#shap.summary_plot(shap_values, X1_test[0:100], max_display=X1_test[0:100].shape[1], plot_type='bar')
#shap.summary_plot(shap_values, X1_test[0:100], max_display=X1_test[0:100].shape[1])
#shap.decision_plot(ex_s.expected_value, shap_values, X1_test.columns)
#shap.plots.force(ex_s.expected_value, shap_values[3, :], X1_test.iloc[3, :], matplotlib = True, figsize=(3, 20))

"""
#LIME
explainer_l = lime.lime_tabular.LimeTabularExplainer(
    X1_train.values,
    feature_names=list(X1_train.columns),
    class_names=['0','1'],
    mode = 'classification',
    random_state = 42,
)

ex_l = explainer_l.explain_instance(X1_test.values[11], lime_tabnet_predict, num_features=4) #X1_test.values[3]
ex_l.save_to_file('amperedata_results/tabnet_b50e50_lime_local_sample11_phase1.html')
figure=ex_l.as_pyplot_figure()
plt.show()
"""

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#NODE

#node_tabular_model.fit(train=data1_train, validation=data1_val)
#node_tabular_model.save_model("smodels/node_b50_e50_phase1_normalized", inference_only=True)
node_tabular_model=TabularModel.load_model("smodels/node_b50_e50_phase1_normalized")

def shap_node_predict(Test):
	global node_tabular_model
	return node_tabular_model.predict(Test)[['prediction']]#.to_numpy()

def lime_node_predict(Test):
	global node_tabular_model
	return node_tabular_model.predict(pd.DataFrame(Test, columns=['I1','P1','Q1','V1']))[['0_probability','1_probability']].to_numpy()

#print(node_tabular_model)

#pred_df_node = node_tabular_model.predict(data1_test, progress_bar="rich")
#eval_result = node_tabular_model.evaluate(data1_test)
#node_pred = pred_df_node['prediction']
#print(node_pred.to_numpy())
#print(y_test.to_numpy().flatten())
#print(classification_report(y1_test.to_numpy().flatten(), node_pred.to_numpy()))

#plt.rcParams['figure.dpi'] = 300
#SHAP
#ex_s = shap.KernelExplainer(shap_node_predict, X1_test[0:100], keep_index=True)
#shap_values = ex_s.shap_values(X1_test[0:100])
#print("Shap Values Shape: ", shap_values.shape)	
#print("Shap Values Shape: ", shap_values[:,:,2].shape)
#print("Expected Value: ", ex_s.expected_value)
#shap.summary_plot(shap_values, X1_test[0:100], max_display=X1_test[0:100].shape[1], plot_type='bar')
#shap.summary_plot(shap_values, X1_test[0:100], max_display=X1_test[0:100].shape[1])
#shap.decision_plot(ex_s.expected_value, shap_values, X1_test.columns)
#shap.plots.force(ex_s.expected_value, shap_values[3, :], X1_test.iloc[3, :], matplotlib = True, figsize=(3, 20))

"""
#LIME
explainer_l = lime.lime_tabular.LimeTabularExplainer(
    X1_train.values,
    feature_names=list(X1_train.columns),
    class_names=['0','1'],
    mode = 'classification',
    random_state = 42,
)

ex_l = explainer_l.explain_instance(X1_test.values[3], lime_node_predict, num_features=4) #X1_test.values[3]
ex_l.save_to_file('amperedata_results/node_b50e50_lime_local_sample11_phase1.html')
figure=ex_l.as_pyplot_figure()
plt.show()
"""

#--------------------------------------------------------------------------------------------------------------
# Random Forest Classifier ()
"""
rf_model = RandomForestClassifier(n_estimators=2, random_state=1)

rf_model.fit(X1_train, y1_train)
rf_pred = rf_model.predict(X1_test)
rf_pred = rf_pred.flatten()
#print(type(rf_pred))
#print(type(y_test.to_numpy().flatten()))

print(classification_report(y1_test.to_numpy().flatten(), rf_pred))
"""
#--------------------------------------------------------------------------------------------------------------
#SVM Classifier 
"""
SVCC = make_pipeline(StandardScaler(), SVC(gamma='auto'))

SVCC.fit(X1_train, y1_train)
svc_pred = SVCC.predict(X1_test)
svc_pred = svc_pred.flatten()
#y_test_np = y_test.to_numpy().flatten()

#print(model_predictions_np.shape)

print(classification_report(y1_test.to_numpy().flatten(), svc_pred))
"""
#--------------------------------------------------------------------------------------------------------------
# XGBoost Classifier
"""
xgb_model = XGBClassifier(n_estimators=2, learning_rate=1, objective='binary:logistic')
xgb_model.fit(X1_train, y1_train)
xgb_pred = xgb_model.predict(X1_test)
xgb_pred = xgb_pred.flatten()

print(classification_report(y1_test.to_numpy().flatten(), xgb_pred))
"""
#--------------------------------------------------------------------------------------------------------------


























# Copyright (C) [2023] [Dr. Pattipong Wisanpitayakorn]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import os, sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Ensure that the script's current working directory is set to the directory where the script itself resides
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

############## Training ##############
# Read the excel file for training
data = pd.read_excel('Training.xlsx', sheet_name=0, engine='openpyxl')
# Set parameters for hyperparameter tuning
HT_params = {'kernel': ['rbf'], 'C': [100, 1000, 10000, 100000], 'gamma': [0.01, 0.1, 1, 10], 'epsilon': [0.01, 0.1, 1]}

CCS = data['Exp CCS'] # Assign variable for training CCS
features = data[['m/z', 'Polarizability']] # Assign training features to be m/z and polarizability columns of the file

## Perform scaling on training data (MinMax or StandardScaler)
# scaler = MinMaxScaler()
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Perform hyperparameter tuning
HT_model = SVR()
search = GridSearchCV(HT_model, param_grid=HT_params, scoring='r2', cv=5)
search.fit(features, CCS)
print('best hyperparameter tuning parameters = ', search.best_params_)
print('Mean cross-validated score of the best_estimator = ', search.best_score_)
best_params = search.best_params_
model = search.best_estimator_

#### Model training ####
modelfit = model.fit(features, CCS)

####################################################################

############## Predicting ##############
data_pred_exp = pd.read_excel('Predicting.xlsx', sheet_name=0, engine='openpyxl')  # Read the excel file for predicting
features_pred = data_pred_exp[['m/z', 'Polarizability']]
features_pred = scaler.fit_transform(features_pred)

CCS_pred = modelfit.predict(features_pred)
data_pred_exp['Predicted CCS'] = CCS_pred
data_pred_exp.to_excel('Results.xlsx', index=False)

### If you want to test the prediction performance, change 'cal_MRE' to 1
### In the 'Predicting' Excel file, also place the exp measured CCS values in a column with the header 'Exp CCS'
cal_MRE = 0

if cal_MRE == 1:
    CCS_pred_exp = data_pred_exp['Exp CCS']
    RE = []
    for i in range(0, len(CCS_pred)):
        dif = (CCS_pred[i] - CCS_pred_exp[i]) / CCS_pred_exp[i] * 100
        RE.append(abs(dif))
    print('Predicting MRE = ', np.mean(RE))
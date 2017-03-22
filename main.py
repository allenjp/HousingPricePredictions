# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:17:30 2017

@author: Jeff
@goal: Create a regressor which will analyze existing housing data and use
        that information to predict unknown house prices given that
        house's features
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

# import the data
data_test = pd.read_csv('Datasets/test_data.csv')
data_train = pd.read_csv('Datasets/train_data.csv')

# clean the training data
#check for nan's
data_train.fillna(0, inplace=True)
data_test.fillna(0, inplace=True)

# NOTE: Looking back, building a function to clean the data and
# create categories would've been cleaner, but this works for now

#MSZoning
data_train['MSZoning'] = data_train.MSZoning.astype("category").cat.codes
data_test['MSZoning'] = data_test.MSZoning.astype("category").cat.codes

#Street
data_train['Street'] = data_train.Street.astype("category").cat.codes
data_test['Street'] = data_test.Street.astype("category").cat.codes

#Alley
ordered_alley=['Grvl', 'Pave']
data_train['Alley'] = data_train.Alley.astype("category", ordered=True, categories=ordered_alley).cat.codes
data_test['Alley'] = data_test.Alley.astype("category", ordered=True, categories=ordered_alley).cat.codes

#LotShape
ordered_shapes=['IR3', 'IR2', 'IR1', 'Reg']
data_train['LotShape'] = data_train.LotShape.astype("category", ordered=True, categories=ordered_shapes).cat.codes
data_test['LotShape'] = data_test.LotShape.astype("category", ordered=True, categories=ordered_shapes).cat.codes

#LandContour
data_train['LandContour'] = data_train.LandContour.astype("category").cat.codes
data_test['LandContour'] = data_test.LandContour.astype("category").cat.codes

#Utilities
ordered_utils=['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
data_train['Utilities'] = data_train.Utilities.astype("category", ordered=True, categories=ordered_utils).cat.codes
data_test['Utilities'] = data_test.Utilities.astype("category", ordered=True, categories=ordered_utils).cat.codes

#LotConfig
data_train['LotConfig'] = data_train.LotConfig.astype("category").cat.codes
data_test['LotConfig'] = data_test.LotConfig.astype("category").cat.codes

#LandSlope
data_train['LandSlope'] = data_train.LandSlope.astype("category").cat.codes
data_test['LandSlope'] = data_test.LandSlope.astype("category").cat.codes

# Calculations for neighborhood to give an order -
# We will find the mean sale price of each neighborhood to
# create an order of least > most expensive
ordered_nhoods = []
mean_sale_prices = {}

for nhood in data_train['Neighborhood'].unique():
    mean_sale_prices[nhood] = data_train[data_train.Neighborhood == nhood].SalePrice.mean()

mean_sale_prices = sorted(mean_sale_prices.items(), key=lambda x: x[1])

for nhood in mean_sale_prices:
    ordered_nhoods.append(nhood[0])
    
# end neighborhood ordering

# resume cleaning categories:
data_train['Neighborhood'] = data_train.Neighborhood.astype("category", ordered=True, categories=ordered_nhoods).cat.codes
data_test['Neighborhood'] = data_test.Neighborhood.astype("category", ordered=True, categories=ordered_nhoods).cat.codes

#Condition1
data_train['Condition1'] = data_train.Condition1.astype("category").cat.codes
data_test['Condition1'] = data_test.Condition1.astype("category").cat.codes

#Condition2
data_train['Condition2'] = data_train.Condition2.astype("category").cat.codes
data_test['Condition2'] = data_test.Condition2.astype("category").cat.codes

#BldgType
data_train['BldgType'] = data_train.BldgType.astype("category").cat.codes
data_test['BldgType'] = data_test.BldgType.astype("category").cat.codes

#HouseStyle
ordered_styles = ['SFoyer', 'SLvl', '1Story', '1.5Unf', '1.5Fin', '2Story', '2.5Unf', '2.5Fin']
data_train['HouseStyle'] = data_train.HouseStyle.astype("category", ordered=True, categories=ordered_styles).cat.codes
data_test['HouseStyle'] = data_test.HouseStyle.astype("category", ordered=True, categories=ordered_styles).cat.codes

#RoofStyle
data_train['RoofStyle'] = data_train.RoofStyle.astype("category").cat.codes
data_test['RoofStyle'] = data_test.RoofStyle.astype("category").cat.codes

#RoofMatl
data_train['RoofMatl'] = data_train.RoofMatl.astype("category").cat.codes
data_test['RoofMatl'] = data_test.RoofMatl.astype("category").cat.codes

#Exterior1st
data_train['Exterior1st'] = data_train.Exterior1st.astype("category").cat.codes
data_test['Exterior1st'] = data_test.Exterior1st.astype("category").cat.codes

#Exterior2nd
data_train['Exterior2nd'] = data_train.Exterior2nd.astype("category").cat.codes
data_test['Exterior2nd'] = data_test.Exterior2nd.astype("category").cat.codes

#MasVnrType
data_train['MasVnrType'] = data_train.MasVnrType.astype("category").cat.codes
data_test['MasVnrType'] = data_test.MasVnrType.astype("category").cat.codes

#ExterQual
ordered_qual = ["Po", "Fa", "TA", "Gd", "Ex"]
data_train['ExterQual'] = data_train.ExterQual.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['ExterQual'] = data_test.ExterQual.astype("category", ordered=True, categories=ordered_qual).cat.codes

#ExterCond
data_train['ExterCond'] = data_train.ExterCond.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['ExterCond'] = data_test.ExterCond.astype("category", ordered=True, categories=ordered_qual).cat.codes

#Foundation
data_train['Foundation'] = data_train.Foundation.astype("category").cat.codes
data_test['Foundation'] = data_test.Foundation.astype("category").cat.codes

#BsmtQual
data_train['BsmtQual'] = data_train.BsmtQual.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['BsmtQual'] = data_test.BsmtQual.astype("category", ordered=True, categories=ordered_qual).cat.codes

#BsmtCond
data_train['BsmtCond'] = data_train.BsmtCond.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['BsmtCond'] = data_test.BsmtCond.astype("category", ordered=True, categories=ordered_qual).cat.codes

#BsmtExposure
ordered_expo = ["No", "Mn", "Av", "Gd"]
data_train['BsmtExposure'] = data_train.BsmtExposure.astype("category", ordered=True, categories=ordered_expo).cat.codes
data_test['BsmtExposure'] = data_test.BsmtExposure.astype("category", ordered=True, categories=ordered_expo).cat.codes


#BsmtFinType1
ordered_fintypes = ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
data_train['BsmtFinType1'] = data_train.BsmtFinType1.astype("category", ordered=True, categories=ordered_fintypes).cat.codes
data_test['BsmtFinType1'] = data_test.BsmtFinType1.astype("category", ordered=True, categories=ordered_fintypes).cat.codes

#BsmtFinType2
data_train['BsmtFinType2'] = data_train.BsmtFinType2.astype("category", ordered=True, categories=ordered_fintypes).cat.codes
data_test['BsmtFinType2'] = data_test.BsmtFinType2.astype("category", ordered=True, categories=ordered_fintypes).cat.codes

#Heating
data_train['Heating'] = data_train.Heating.astype("category").cat.codes
data_test['Heating'] = data_test.Heating.astype("category").cat.codes

#HeatingQC
data_train['HeatingQC'] = data_train.HeatingQC.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['HeatingQC'] = data_test.HeatingQC.astype("category", ordered=True, categories=ordered_qual).cat.codes

#CentralAir
ordered_yn = ['N', 'Y']
data_train['CentralAir'] = data_train.CentralAir.astype("category", ordered=True, categories=ordered_yn).cat.codes
data_test['CentralAir'] = data_test.CentralAir.astype("category", ordered=True, categories=ordered_yn).cat.codes

#Electrical
data_train['Electrical'] = data_train.Electrical.astype("category").cat.codes
data_test['Electrical'] = data_test.Electrical.astype("category").cat.codes

#KitchenQual
data_train['KitchenQual'] = data_train.KitchenQual.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['KitchenQual'] = data_test.KitchenQual.astype("category", ordered=True, categories=ordered_qual).cat.codes

#Functional
ordered_fun = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
data_train['Functional'] = data_train.Functional.astype("category", ordered=True, categories=ordered_fun).cat.codes
data_test['Functional'] = data_test.Functional.astype("category", ordered=True, categories=ordered_fun).cat.codes

#FireplaceQu
data_train['FireplaceQu'] = data_train.FireplaceQu.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['FireplaceQu'] = data_test.FireplaceQu.astype("category", ordered=True, categories=ordered_qual).cat.codes

#GarageType
garage_orders = ["Detchd", "CarPort", "BuiltIn", "Basment", "Attchd", "2Types"]
data_train['GarageType'] = data_train.GarageType.astype("category", ordered=True, categories=garage_orders).cat.codes
data_test['GarageType'] = data_test.GarageType.astype("category", ordered=True, categories=garage_orders).cat.codes

#GarageFinish
ordered_garfin = ["Unf", "RFn", "Fin"]
data_train['GarageFinish'] = data_train.GarageFinish.astype("category", ordered=True, categories=ordered_garfin).cat.codes
data_test['GarageFinish'] = data_test.GarageFinish.astype("category", ordered=True, categories=ordered_garfin).cat.codes

#GarageQual
data_train['GarageQual'] = data_train.GarageQual.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['GarageQual'] = data_test.GarageQual.astype("category", ordered=True, categories=ordered_qual).cat.codes

#GarageCond
data_train['GarageCond'] = data_train.GarageCond.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['GarageCond'] = data_test.GarageCond.astype("category", ordered=True, categories=ordered_qual).cat.codes

#PavedDrive
ordered_paves = ["N", "P", "Y"]
data_train['PavedDrive'] = data_train.PavedDrive.astype("category", ordered=True, categories=ordered_paves).cat.codes
data_test['PavedDrive'] = data_test.PavedDrive.astype("category", ordered=True, categories=ordered_paves).cat.codes

#PoolQC
data_train['PoolQC'] = data_train.PoolQC.astype("category", ordered=True, categories=ordered_qual).cat.codes
data_test['PoolQC'] = data_test.PoolQC.astype("category", ordered=True, categories=ordered_qual).cat.codes

#Fence
ordered_fence = ["MnWw", "GdWo", "MnPrv", "GdPrv"]
data_train['Fence'] = data_train.Fence.astype("category", ordered=True, categories=ordered_fence).cat.codes
data_test['Fence'] = data_test.Fence.astype("category", ordered=True, categories=ordered_fence).cat.codes

#MiscFeature
data_train['MiscFeature'] = data_train.MiscFeature.astype("category").cat.codes
data_test['MiscFeature'] = data_test.MiscFeature.astype("category").cat.codes

#SaleType
data_train['SaleType'] = data_train.SaleType.astype("category").cat.codes
data_test['SaleType'] = data_test.SaleType.astype("category").cat.codes

#SaleCondition
data_train['SaleCondition'] = data_train.SaleCondition.astype("category").cat.codes
data_test['SaleCondition'] = data_test.SaleCondition.astype("category").cat.codes

# Before work on prediction algorithms, visualize the data first to 
# get a better feel for it

# histogram of prices so we get a general feel for how they are spread
pd.DataFrame.hist(data_train[['SalePrice']])

# the above chart seems to indicate two outliers with very high prices -
# we will drop those two houses from the training data before proceeding
data_train = data_train[data_train['SalePrice'] < 700000]

# how house quality compares to sale price
data_train.plot.scatter(x='OverallQual', y='SalePrice')
# we see a positive correlation here, which makes sense

# lastly we will do a 3d scatter plot comparing lot area, number of rooms, and sale price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Lot Area')
ax.set_ylabel('Number of Rooms')
ax.set_zlabel('SalePrice')

# ax.scatter(data_train.LotArea, data_train.TotRmsAbvGrd, data_train.SalePrice, c='r', marker='.')

# some outliers skew the visualization - rather than dropping them I will create a subset
# of the data to hopefully obtain a clearer picture
temp_data = data_train[data_train['LotArea'] < 100000]
ax.scatter(temp_data.LotArea, temp_data.TotRmsAbvGrd, temp_data.SalePrice, c='r', marker='.')

# it's clear that number of rooms has a positive correlation to price, 
# but the relationship with lot area is still a bit hazy

# show the plots
plt.show()

# separate labels from training data for use in RandomForest
label_train = data_train['SalePrice'].copy()
data_train.drop(labels=['SalePrice'], inplace=True, axis=1)

# init RandomForestRegressor which we will use to predict the test data
model = RandomForestRegressor(n_estimators=500, max_features="auto", max_depth=20)

#
# TODO: train your model on your training set
#
model.fit(data_train, label_train)

# Do predictions here
pred_prices = model.predict(data_test)

# write to csv with results
pred_ids = data_test.Id

result = pd.DataFrame({'Id': pred_ids,
                       'SalePrice': pred_prices})

result.to_csv('Datasets/submission.csv')

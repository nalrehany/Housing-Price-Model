import numpy as np  
import matplotlib.pyplot as plt # .pyplot is pyplot module of matplotlib
import pandas as pd 
from sklearn.impute import SimpleImputer
import math
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

### Import and do some cleaning ###

hamiltondata = pd.read_csv("C:/Users/Nick/Downloads/hamiltondata.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
hamiltondata.drop(['CTUID','drivetowork', 'bachdegree', 'manager', 'single-detached','fourplusrooms', 'majorrepairs', 'workfromhome', 'occupieddwellings', 'totalcommuter', 'totaleducated', 'labourforce', 'employed'], axis=1, inplace=True)
hamiltondata[['PctSingleDetached', 'PctDrivetoWork', 'PctBachelorDegree', 'PctofManagers', 'PctFourPlusRooms', 'PctNeedMajorRepair', 'PctWorkFromHome']] = hamiltondata[['PctSingleDetached', 'PctDrivetoWork', 'PctBachelorDegree', 'PctofManagers', 'PctFourPlusRooms', 'PctNeedMajorRepair', 'PctWorkFromHome']]*100

### Precheck of linearity of variables using a sefl-made function that displays scatter plots if wanting to manually select variables ###

def plot(df, dependent, independent):
	plt.scatter(df[independent], df[dependent], color='red')
	plt.title(f'{dependent} Vs {independent}', fontsize=14)
	plt.xlabel([independent], fontsize=14)
	plt.ylabel([dependent], fontsize=14)
	plt.grid(True)
	plt.show()

plot(hamiltondata, 'AvgHousePrice', 'avgincome')
plot(hamiltondata, 'AvgHousePrice', 'PctSingleDetached')
plot(hamiltondata, 'AvgHousePrice', 'PctDrivetoWork')
plot(hamiltondata, 'AvgHousePrice', 'PctBachelorDegree')
plot(hamiltondata, 'AvgHousePrice', 'PctofManagers')
plot(hamiltondata, 'AvgHousePrice', 'employmentrate')
plot(hamiltondata, 'AvgHousePrice', 'PctFourPlusRooms')
plot(hamiltondata, 'AvgHousePrice', 'PctNeedMajorRepair')
plot(hamiltondata, 'AvgHousePrice', 'avgage')
plot(hamiltondata, 'AvgHousePrice', 'PctWorkFromHome')


### Creating dependent and independent variables based on most linearly related independent variables ###

X = hamiltondata[['avgincome', 'PctBachelorDegree', 'PctFourPlusRooms']].values
y = hamiltondata.iloc[:, -1].values

print(X)
print(y)

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # This is an object of the imputer class. It will help us find that average to infer. 
                         # Instructs to find missing and replace it with mean

#Fit method in SimpleImputer will connect imputer to our matrix of features                       
imputer.fit(X[:,:]) # We exclude column "O" AKA Country because they are strings
X[:, :] = imputer.transform(X[:,:])


# Remove outliers if necessary #
# z = np.abs(stats.zscore(X))
# mask = (z < 3).all(axis=1)
# X = X[mask]
# y = y[mask]

# ## Splitting into training and testing ##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# ### Feature Scaling ###

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Standardize variables
X_train[:, 0:] = sc.fit_transform(X_train[:,0:])
# Fit changes the data, Transform applies it! Here we have a method that does both

X_test[:, 0:] = sc.transform(X_test[:, 0:]) 

print(X_train)
print(X_test)

## Training ## 
from sklearn.linear_model import LinearRegression 

regressor = LinearRegression() 
model = regressor.fit(X_train, y_train)

### Predicting Test Set results ###

y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2) # Display any numerical value with only 2 numebrs after decimal
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1 )), axis=1)) # this just simply makes everything vertical


# Model Summary #
model = sm.OLS(Y_train,X_train)

result=model.fit()

print(result.summary())

# R-squared = 0.91
# Adjusted R-squared = 0.72


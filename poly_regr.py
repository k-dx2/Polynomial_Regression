import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#reading the csv file
df = pd.read_csv("FuelConsumption.csv")
# take a look at the dataset
print(df.head())
# summarize the data
print(df.describe())


#selecting features for Linear regression and spliting it in test(30%) and train data set
X= df[['ENGINESIZE']]
y=df[['CO2EMISSIONS']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#applying Polynomial regression between enginesize and CO2 emission
poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(X_train)
print(train_x_poly)


clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, y_train)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)
print ('\n')


#plotting the data 
plt.scatter(X.ENGINESIZE, y.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy,'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


#testing metrics
test_x_poly = poly.fit_transform(X_test)
test_y_hat=clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat,y_test) )




# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Negative Skew"])

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew_1'],line='45')
plt.show()

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Positive Skew_1'],line='45')
plt.show()

df
```

# OUPUT
![image](https://user-images.githubusercontent.com/98682825/174452473-1b9965cf-ad37-4cda-827b-77bd8542986c.png)
![image](https://user-images.githubusercontent.com/98682825/174452496-416ad975-4f0a-492a-9bc6-26cc4d88223a.png)
![image](https://user-images.githubusercontent.com/98682825/174452513-c44d7931-250a-4b1e-b30a-e43b8159b425.png)
![image](https://user-images.githubusercontent.com/98682825/174452555-5d16a61c-0a58-47b3-8fc8-010cbe777dfc.png)
![image](https://user-images.githubusercontent.com/98682825/174452587-e04b6e41-ecc0-4b57-8295-32badb1817f1.png)
![image](https://user-images.githubusercontent.com/98682825/174452595-c043dfb3-3e59-4b97-bab9-2eedcdb2080c.png)
![image](https://user-images.githubusercontent.com/98682825/174452604-a2b37a40-2579-4526-90fc-c473e5c8f1f2.png)
![image](https://user-images.githubusercontent.com/98682825/174452613-2927ad9f-905c-4e70-b09f-db16fb5a86b7.png)
![image](https://user-images.githubusercontent.com/98682825/174452621-0b5b56e7-2e2f-4d41-b196-0c5ffecebae4.png)
![image](https://user-images.githubusercontent.com/98682825/174452665-23a314a1-e158-4782-8e0d-479a29b09c60.png)
![image](https://user-images.githubusercontent.com/98682825/174452677-8e0196a1-30e0-41cb-91ca-b526f05d9415.png)
![image](https://user-images.githubusercontent.com/98682825/174452694-152e1559-e5d6-4e07-bc4a-0febed31a901.png)
![image](https://user-images.githubusercontent.com/98682825/174452701-d73ba122-0e9b-47fc-980e-2c27724184aa.png)
![image](https://user-images.githubusercontent.com/98682825/174452710-83863265-4d60-455f-835e-2cbac388aace.png)
![image](https://user-images.githubusercontent.com/98682825/174452719-ea54521e-fe7a-4fa9-a42c-77b019384ef7.png)
![image](https://user-images.githubusercontent.com/98682825/174452731-be1d5c32-07dc-435f-9cee-e22fde653631.png)
![image](https://user-images.githubusercontent.com/98682825/174452741-9a569241-f7a5-4af7-938a-4257f191f783.png)
![image](https://user-images.githubusercontent.com/98682825/174452749-02e42360-809c-4775-b054-002a3a28175e.png)
![image](https://user-images.githubusercontent.com/98682825/174452754-a3f6445b-7e1a-4ce0-8bf8-498c4fed7234.png)
![image](https://user-images.githubusercontent.com/98682825/174452760-a1e34de3-8168-47e1-98a1-80471ff28b22.png)








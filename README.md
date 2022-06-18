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


![image](https://user-images.githubusercontent.com/98682825/174452821-e3f6ae99-bb39-4309-a462-d93dba2f61da.png)
![image](https://user-images.githubusercontent.com/98682825/174452825-9f64e7bd-d6ad-47b0-a4b5-03db153c363d.png)
![image](https://user-images.githubusercontent.com/98682825/174452828-737c73e0-1e33-4381-ab9c-5480340753ab.png)
![image](https://user-images.githubusercontent.com/98682825/174452836-56d39f74-b18f-4160-9a18-3cd8bb2baf3e.png)
![image](https://user-images.githubusercontent.com/98682825/174452838-37240eee-56e6-4d19-84ca-c11191603b29.png)
![image](https://user-images.githubusercontent.com/98682825/174452842-5f49050b-39fa-44aa-85ea-757e5e7a1cb3.png)
![image](https://user-images.githubusercontent.com/98682825/174452849-d931a611-4151-4259-bfa1-44730042875d.png)
![image](https://user-images.githubusercontent.com/98682825/174452853-d6dd0e46-939d-4d6c-9080-594139f69d08.png)
![image](https://user-images.githubusercontent.com/98682825/174452861-0ebe2379-bbcf-4596-8809-e6eacc0a4a15.png)
![image](https://user-images.githubusercontent.com/98682825/174452864-e9152876-383a-41d8-8e2a-f982bc4289b9.png)
![image](https://user-images.githubusercontent.com/98682825/174452869-d9f1f5b8-3045-461b-903b-f8c138ed549d.png)
![image](https://user-images.githubusercontent.com/98682825/174452872-9f266457-71db-420b-a95d-752a7fc821b7.png)
![image](https://user-images.githubusercontent.com/98682825/174452879-5ddfdf51-37a3-4d8a-9763-28796945b4a7.png)
![image](https://user-images.githubusercontent.com/98682825/174452880-4af6a4dd-6917-46e4-a297-6605ff6f8c7c.png)
![image](https://user-images.githubusercontent.com/98682825/174452884-5ca363c6-4289-424b-8764-5a4bf6022574.png)
![image](https://user-images.githubusercontent.com/98682825/174452900-c10d3874-6ebe-412f-9ae5-0bb5157840f0.png)
![image](https://user-images.githubusercontent.com/98682825/174452902-30369628-1756-4f46-97a0-ac71a670d247.png)
![image](https://user-images.githubusercontent.com/98682825/174452910-5cba3c61-4d2a-40fb-ae12-73d09fc3788f.png)
![image](https://user-images.githubusercontent.com/98682825/174452913-c51e9747-39d5-4263-a7a7-a4aa0c7d279f.png)
![image](https://user-images.githubusercontent.com/98682825/174452919-aa941de7-adf2-45b8-9652-9a50d22c21ba.png)
![image](https://user-images.githubusercontent.com/98682825/174452927-e496f512-92c0-47c9-a267-9a516d4a1337.png)









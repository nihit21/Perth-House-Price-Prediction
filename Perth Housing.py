#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all necessary libraries including the basics loibraries used for visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing the dataset
df1 = pd.read_csv("all_perth_310121.csv") 
df1 = pd.DataFrame(df1)


# In[3]:


df1.head(10)


# In[4]:


df1.shape


# In[5]:


#to get the description of the data
df1.describe()


# In[6]:


df1.info()


# In[7]:


df1.sort_values(by = ['PRICE'] , ascending = False)


# In[8]:


sns.histplot(data = df1 , x = 'PRICE', bins = 50)


# In[9]:


#finding out null values
df1.isna().any()


# In[10]:


#we can observe that in Garage , Build year , Nearest_sch_rank have null values so will deal with the values.
sns.heatmap(df1.isnull() , cbar = False , cmap="viridis")


# In[11]:


#to get the percentage of missing values so can eliminate the columns which have higher than 25% of the total null values
df1.isnull().mean()*100


# In[12]:


df_clean = df1 #so that no alteraton are in the initial dataframes


# In[13]:


#Cleaning the data
#dropping Address as has high cardinality
df_clean.drop("ADDRESS", axis = 1 , inplace = True)


# In[14]:


#dealing with null values in Garage columnxx
sns.boxenplot(data = df_clean ,x = 'GARAGE')


# In[15]:


sns.distplot(df_clean['GARAGE'])


# In[16]:


df_clean['GARAGE'] = df_clean['GARAGE'].fillna(df_clean['GARAGE'].mean())


# In[17]:


#now for BUILD-_YEAR column
sns.distplot(df_clean['BUILD_YEAR'], bins =10)


# In[18]:


#Mean values to fll thenull values for the build year 
df_clean['BUILD_YEAR'] = df_clean['BUILD_YEAR'].fillna(df_clean['BUILD_YEAR'].mean())


# In[19]:


#so dropping NEAREST_SCH_RANK because  32.541003 > 25% which can effect our model
df_clean.drop('NEAREST_SCH_RANK', axis =1 , inplace = True)


# In[20]:


df_clean.isna().any()


# In[21]:


#plotting on heatmap 
sns.heatmap(df_clean.isnull())
#As now there are no null values


# In[22]:


df_clean.head()


# In[23]:


df_clean.dtypes


# In[24]:


#changing date_sold column to date type 
df_clean['DATE_SOLD'] = df_clean['DATE_SOLD'].astype('datetime64[ns]')


# In[25]:


df_clean.info()


# In[26]:


df_clean1 = df_clean


# In[27]:


#As we do not need SUBURB , Nearest_stn and Nearest_schl as we have respective columns like pincode, NEAREST_STN_DIST , NEAREST_SCH_D
#IST which can give us the idea of the location so we will drop them.
df_clean1.drop(['SUBURB', 'NEAREST_STN', 'NEAREST_SCH'] , axis = 1, inplace = True)


# In[28]:


print(f"maximum number of bedrooms = {df_clean['BEDROOMS'].max()}")
print("\n")
print(f"maximum number of bathroom = {df_clean['BATHROOMS'].max()}")
print("\n")
print(f"maximum number of garage = {df_clean['GARAGE'].max()}")


# In[29]:


#Not possible house with 10 bedrooms having 99 garages and 12 washrooms so will filter the data that having maximum 5 Garages
df_new = df_clean1
df_new = df_new.drop(df_new[df_new["GARAGE"] > 5].index)
df_new = df_new.drop(df_new[df_new['BATHROOMS'] >10].index)


# In[30]:


#Exploring effects of different factors on Price

#Number bedroom vs price
plt.figure(figsize=(10,8))
sns.lineplot(x = 'BEDROOMS', y = 'PRICE', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,10))
plt.title("Bedrooms vs Price")


# In[31]:


#no of bathroom vs price
plt.figure(figsize =(14,8))
sns.barplot(x = 'BATHROOMS', y = 'PRICE', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,7))
plt.title("Number of bathroom's vs Price")


# In[32]:


#no of Garages vs Price
plt.figure(figsize =(14,8))
sns.barplot(x = 'GARAGE', y = 'PRICE', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,7))
plt.title("Number of Garage vs Price")


# In[33]:


#plot for Bedroom vs floor area
plt.figure(figsize=(12,10))
sns.violinplot(x = 'BEDROOMS', y = 'FLOOR_AREA', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(3,10))
plt.title("Bedrooms vs Floor area")


# In[34]:


#plot for Bathroom vs floor area
plt.figure(figsize=(12,10))
sns.violinplot(x = 'BATHROOMS', y = 'FLOOR_AREA', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(3,10))
plt.title("Bathrooms vs Floor Area")


# In[35]:


#Garage vs Floor Area
plt.figure(figsize=(12,10))
sns.violinplot(x = 'GARAGE', y = 'FLOOR_AREA', data = df_new)
plt.ticklabel_format(style='sci', axis='y', scilimits=(3,10))
plt.title("GARAGE vs Floor Area")


# In[36]:


#Taking idea about dispersion of houses
plt.figure(figsize=(10,8))
sns.jointplot(x = 'LATITUDE', y = 'LONGITUDE', data = df_new)
plt.suptitle("Geographical spread of the houses")


# In[37]:


#Dispersion of price
sns.histplot(df_new["PRICE"], bins=100)
plt.title("Prices")
plt.xlim([50000, 2500000])
plt.xticks(range(50000, 2500000, 500000), ["50K", "550K", "1.05M", "1.55M", "2.05M"])
plt.xlabel("Prices of houses")


# In[ ]:





# In[38]:


#pairplot for selective columns
sns.pairplot(df_new[['PRICE','BEDROOMS', 'FLOOR_AREA', 'CBD_DIST']])


# In[39]:


#heatmap showing cor relation between different variables to get better understanding for the supervised learning.
plt.figure(figsize=(10,8))
sns.heatmap(df_new.corr(), annot = True)
plt.title("Correlation matrix")
df_new.corr()


# In[40]:


#floor area vs price
plt.figure(figsize = (12,10))
sns.scatterplot(x =  'FLOOR_AREA', y = 'PRICE', data = df_new )
plt.yticks(range(50000,2500000,500000), ["50K", "550K", "1.05M", "1.55M", "2.05M"])
plt.title("Floor Size Vs Price")


# In[41]:


df_clean2 = df_new


# In[42]:


df_clean2.drop("DATE_SOLD", axis = 1, inplace= True)

Supervised learning

# In[43]:


#making X and y variables for further analysis
X = df_clean2.drop(["PRICE"] ,axis =1 )
X.head()


# In[44]:


y = df_clean2['PRICE']


# In[45]:


y.head()


# In[46]:


#Test and train data split
from sklearn.model_selection import train_test_split


# In[47]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=101)


# In[48]:


#Linear regression
from sklearn.linear_model import LinearRegression


# In[49]:


#fitting data into linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[50]:


#evaluating model
#lm.intercept tells the mean value of the response variable when all of the predictor variables in the model are equal to zero.

print(lm.intercept_)

#coef_ contain the coefficients for the prediction of each of the targets.
lm.coef_


# In[51]:


#dataframe for plotting coef with columns
c_df = pd.DataFrame(lm.coef_, X.columns , columns = ['Coeff'])
c_df.head()
#it shows if all the factors are freezed and onlythe respective variable changes than how much it effect the y(Price) vaiable


# In[52]:


#getting prediction from the test set
prediction1 = lm.predict(X_test)


# In[53]:


#to evaluate the model that how good it is 
sns.regplot(x = y_test, y =prediction1,ci = None , marker = '.')
plt.title('Linear regression plot')


# In[54]:


sns.distplot((y_test-prediction1),bins=50)


# In[55]:


from sklearn import metrics


# In[56]:


#calculating mean squared eror,Mean squared eror, root mean squared eror,
print('MAE:', metrics.mean_absolute_error(y_test, prediction1))
print('MSE:', metrics.mean_squared_error(y_test, prediction1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction1)))


# In[57]:


#r2 score
from sklearn.metrics import r2_score
r2_score( y_test, prediction1)


# In[58]:


#Results for logistic regression are not so trying Decision Tree and Random Forest .


# In[59]:


#Decision Tree
from sklearn.tree import DecisionTreeRegressor


# In[60]:


dtree = DecisionTreeRegressor()


# In[61]:


dtree.fit(X_train, y_train)


# In[62]:


prediction2 = dtree.predict(X_test)


# In[63]:


#evaluating the model
plt.scatter(y_test, prediction2)


# In[64]:


sns.distplot((y_test-prediction2),bins=50)


# In[65]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction2))
print('MSE:', metrics.mean_squared_error(y_test, prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction2)))


# In[66]:


#r2 score for Decision tree
r2_score( y_test, prediction2)


# In[67]:


#Using random forests to get best result as if we observe scatter plot points are not at centre 
from sklearn.ensemble import RandomForestRegressor


# In[68]:


rfc = RandomForestRegressor(n_estimators=50)


# In[69]:


rfc.fit(X_train,y_train)


# In[70]:


rfc_predict = rfc.predict(X_test)


# In[71]:


#evaluating the model
sns.regplot(x = y_test, y = rfc_predict, ci = None, marker = ".")
plt.title("Random forest regressor")


# In[72]:


print('MAE:', metrics.mean_absolute_error(y_test, rfc_predict))
print('MSE:', metrics.mean_squared_error(y_test, rfc_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfc_predict)))


# In[73]:


r2_score(y_test, rfc_predict)


# UNSUPERVISED

# In[74]:


#importing Librariy for KNN 
from sklearn.cluster import KMeans


# In[75]:


df_clean2.info()


# In[76]:


#Scaling features in the dataset from 0 to 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_clean2.drop("PRICE", axis =1))


# In[77]:


scaled_features = scaler.transform(df_clean2.drop('PRICE', axis =1 ))


# In[78]:


#making the data frame with scaled features
df_features = pd.DataFrame(scaled_features , columns = df_clean2.columns[1:])


# In[79]:


df_features.head()


# In[80]:


##Giving a range for k-means, so that I can determine the number of clusters to go for optimally 
#Also forming an empty list for error_rate to capture standard error 
error_rate = []
for i in range(1,50):
    knn = KMeans(n_clusters= i)
    knn.fit(df_features)
    error_rate.append(knn.inertia_)


# In[81]:


error_rate


# In[82]:


plt.figure(figsize = (12,6))
plt.plot(range(1,50), error_rate, color = 'blue' , marker = '*', linestyle = '--', markersize = 15)
plt.title('Elbow Curve')
plt.xlabel('Clusters')
plt.ylabel('Error_rate')


# In[83]:


#As we can observe in Elbow curve suitable cluster should be 10
km = KMeans(n_clusters= 10)
y_predicted = km.fit_predict(df_features)


# In[84]:


df_features['cluster'] = y_predicted


# In[85]:


km.cluster_centers_


# In[86]:


##Calculating silhouette score
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score


# In[87]:


score = silhouette_score(df_features, km.labels_, metric='euclidean')
score


# In[88]:


#homogenity score
homogeneity_score(df_new['PRICE'], y_predicted)


# In[90]:


#completness score
completeness_score(df_new['PRICE'], y_predicted)


# In[91]:


sns.scatterplot(data=df_features, x="FLOOR_AREA", y="NEAREST_SCH_DIST", hue="cluster")
plt.ylabel("Distance to nearest school")
plt.xlabel("Floor area")


# In[92]:


#make one more plot 
sns.scatterplot(data = df_features , x = 'FLOOR_AREA' , y ='NEAREST_STN_DIST', hue = 'cluster' )
plt.ylabel("Nearest Station Distance")
plt.xlabel("Floor Area")


# In[93]:


sns.scatterplot(data = df_features , x = 'FLOOR_AREA' , y ='CBD_DIST', hue = 'cluster' )
plt.ylabel("Distance to business hub")
plt.xlabel("Floor Area")


# In[ ]:





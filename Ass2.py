

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[30]:


Zomato=pd.read_excel('zomato_train.xlsx')


# In[31]:


Zomato.isnull().sum()


# In[3]:


zomato=pd.read_excel('zomato_train.xlsx',na_values=['-'])
zomato


# In[ ]:





# In[ ]:





# In[4]:


ZOMATO=zomato.copy()


# In[29]:


ZOMATO.isnull().sum()


# In[76]:


ZOMATO.describe()


# In[77]:


ZOMATO['location'].nunique()


# In[78]:


ZOMATO['location'].unique()


# In[79]:


ZOMATO[(ZOMATO['online_order'] == "Yes")]


# In[5]:


#making frequency tables
pd.crosstab(index=ZOMATO['rates'],columns='count',dropna=True)


# In[ ]:





# In[6]:


pd.crosstab(index=ZOMATO['location'],columns='count',dropna=True)


# In[7]:


#comparison of rates w.r.t location
pd.crosstab(index=ZOMATO['location'],columns=ZOMATO['rates'],dropna=True)


# In[8]:


#comparison of rates w.r.t approx cost(for two people)
pd.crosstab(index=ZOMATO['approx_cost(for two people)'],columns=ZOMATO['rates'],dropna=True)


# In[9]:


#comparison of rates w.r.t location(joint probability)
pd.crosstab(index=ZOMATO['location'],columns=ZOMATO['rates'],normalize=True,dropna=True)


# In[10]:


#comparison of rates w.r.t approx cost(for two people) - (joint probability)
pd.crosstab(index=ZOMATO['approx_cost(for two people)'],columns=ZOMATO['rates'],normalize=True,dropna=True)


# In[11]:


#comparison of rates w.r.t location(marginal probability)
pd.crosstab(index=ZOMATO['location'],columns=ZOMATO['rates'],margins=True,normalize=True,dropna=True)


# In[12]:


#comparison of rates w.r.t approx cost(for two people) - (marginal probability)
pd.crosstab(index=ZOMATO['approx_cost(for two people)'],columns=ZOMATO['rates'],margins=True,normalize=True,dropna=True)


# In[13]:


#comparison of rates w.r.t location(conditional probability)
pd.crosstab(index=ZOMATO['location'],columns=ZOMATO['rates'],margins=True,normalize='index',dropna=True)


# In[14]:


#comparison of rates w.r.t approx cost(for two people) - (conditional probability)
pd.crosstab(index=ZOMATO['approx_cost(for two people)'],columns=ZOMATO['rates'],margins=True,normalize='index',dropna=True)


# In[15]:


ZOMATO['rates']=ZOMATO['rates'].astype('float')


# In[24]:


ZOMATO.info()


# In[23]:


ZOMATO['approx_cost(for two people)']=ZOMATO['approx_cost(for two people)'].astype('int')


# In[19]:


ZOMATO['approx_cost(for two people)'] = ZOMATO['approx_cost(for two people)'].str.replace(',', '')


# In[22]:


ZOMATO=ZOMATO.dropna()


# In[25]:


numerical_data = ZOMATO.select_dtypes(exclude=[object])


# In[26]:


print(numerical_data.shape)


# In[28]:


#Correlation between votes &approx_cost(for two people)& rates
#as the columns 'approx_cost(for two people)' and 'rates' has a positive correlation no. , we can say that as the approx cost increases the rating also increases.
corr_matrix = numerical_data.corr()
corr_matrix


# In[36]:


#Scatter plot - rates vs. approx_cost(for two people) using matplotlib library
plt.scatter(ZOMATO['rates'],ZOMATO['approx_cost(for two people)'],c='red')


# In[57]:


n = 10
loc=ZOMATO['location'].value_counts().head()
loc


# In[55]:


#frerquency bar plot for the top 5 places with most restaurants
counts=[1414,1151,1059,901,825]
location=('Koramangala 5th Block','BTM ','Indiranagar','HSR','Jayanagar')
index=np.arange(len(location))

plt.bar(index,counts)
plt.title('Top 5 places with most Restaurants')
plt.xlabel('Locations')
plt.ylabel('No. of restaurants')
plt.xticks(index,location,rotation=90)
plt.show()


# In[59]:


#Scatter plot - rates vs. approx_cost(for two people) using seaborn library
sns.set(style='darkgrid')
sns.regplot(x=ZOMATO['rates'], y=ZOMATO['approx_cost(for two people)'],marker="*")


# In[62]:


# this histogram shows approx cost distribution
sns.distplot(ZOMATO['approx_cost(for two people)'],kde=False, bins=10)


# In[63]:


#box plot of rates
sns.boxplot(y=ZOMATO['rates'])






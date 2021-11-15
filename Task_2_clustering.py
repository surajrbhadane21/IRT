#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("smooth_theta.csv")
df.head()


# In[3]:


df.info()


# In[4]:


import plotly.express as px
fig = px.scatter_matrix(df.drop('Goal Levels',axis=1),width=1000,height=1000)
fig.show()


# ## Finding our clusters

# In[5]:


from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np


# In[6]:


x = df.drop("Goal Levels",axis=1)
x.head()


# In[7]:


inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, tol=1e-04, random_state=42)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)


# In[ ]:





# In[8]:


fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
fig.update_layout(title="Inertia vs Cluster Number", xaxis=dict(range=[0,11], title = "Cluster Number"),yaxis={'title':'Inertia'},
                  annotations = [dict(
            x=3,
            y=inertia[2],
            xref="x",
            yref="y",
            text="Elbow!",
            showarrow=True,
            arrowhead=7,
            ax=20,
            ay=-40
        )])


# In[9]:


kmeans = KMeans(n_clusters=3, init = 'k-means++', n_init = 10, tol=1e-04, random_state=42)
kmeans.fit(x)


# In[10]:


clusters = pd.DataFrame(x,columns=df.drop('Goal Levels',axis=1).columns)
clusters['label']=kmeans.labels_


# In[11]:


clusters.head()


# In[12]:


polar = clusters.groupby('label').mean().reset_index()
polar


# In[26]:


polar = pd.melt(polar, id_vars=['label'])
polar


# In[14]:


fig4 = px.line_polar(polar, r="value", theta="variable",color="label", line_close=True, height=800, width=800)
fig4.show()


# In[15]:


pie = clusters.groupby('label').size().reset_index()
pie.columns = ['label','value']
px.pie(pie,values='value',names='label',color=['blue','red','green'])


# In[16]:


df1 = clusters.copy()
df1.head()


# In[17]:


df2 = df1[df1['label']==0].reset_index().drop(['index','label'],axis=1)
df2.head()


# In[18]:


df2 = df2.mean().reset_index()
df2.columns=['cohort','means']
df2


# In[19]:


df2['cohort'] = np.arange(1,16)
df2


# In[20]:


fig1 = px.line(df2,x = df2['cohort'],y = df2['means'])
fig1.show()


# ## Polynomial Regression

# In[21]:


x = df2['cohort'].values
y = df2['means'].values


# In[22]:


x=x.reshape(15,1)


# In[23]:


x


# In[24]:


y


# In[377]:


# Split the data into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)


# In[41]:


#Fitting Polynomial Regression Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_linear = LinearRegression()
poly_linear.fit(x_poly,y)


# In[ ]:





# In[31]:


#Visualizing the polynomial Regression
import matplotlib.pyplot as plt

def viz_polynomial():
    plt.scatter(x,y,color='red')
    plt.plot(x,poly_linear.predict(x_poly),color='blue')
    plt.title('Polynomial Regression')
    plt.xlabel('cohort')
    plt.ylabel('menas')
    plt.show()
    return
viz_polynomial()


# In[ ]:





# In[380]:


#Predict a new result with Polynomial Regrssion

poly_linear.predict(poly_reg.fit_transform([[1]]))


# ## For all clusters

# In[46]:


for i in np.sort(df1['label'].unique()):
    df2 = df1[df1['label']==i].reset_index().drop(['index','label'],axis=1)
    df2 = df2.mean().reset_index()
    df2.columns=['cohort','means']
    df2['cohort'] = np.arange(1,16)
    
    #Polynomial Regression
    x = df2['cohort'].values
    x=x.reshape(15,1)
    y = df2['means'].values
    
    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(x)
    poly_linear = LinearRegression()
    poly_linear.fit(x_poly,y)
    
    def viz_polynomial():
        plt.scatter(x,y,color='red')
        plt.plot(x,poly_linear.predict(x_poly),color='blue')
        plt.title('Polynomial Regression')
        plt.xlabel('cohort')
        plt.ylabel('menas')
        plt.show()
        return
    viz_polynomial()    


# ## Function for cohort

# In[47]:


def clusterNum(i):
   '''i = cohort number'''    
   df2 = df1[df1['label']==i].reset_index().drop(['index','label'],axis=1)
   df2 = df2.mean().reset_index()
   df2.columns=['cohort','means']
   df2['cohort'] = np.arange(1,16)
   
   #Polynomial Regression
   x = df2['cohort'].values
   x=x.reshape(15,1)
   y = df2['means'].values
   
   poly_reg = PolynomialFeatures(degree=4)
   x_poly = poly_reg.fit_transform(x)
   poly_linear = LinearRegression()
   poly_linear.fit(x_poly,y)
   
   def viz_polynomial():
       plt.scatter(x,poly_linear.predict(x_poly),color='red')
       plt.plot(x,y,color='blue')
       plt.title('Polynomial Regression')
       plt.xlabel('cohort')
       plt.ylabel('menas')
       plt.show()
       return
   #viz_polynomial()    
   return viz_polynomial()


# In[51]:


clusterNum(2)


# In[ ]:





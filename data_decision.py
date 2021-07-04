#!/usr/bin/env python
# coding: utf-8

# # Project Goal:

# Task: You are an analyst at a big online store. Together with the marketing department, you've compiled a list of hypotheses that may help boost revenue.

# ## Prioritizing Hypotheses 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st


# Imported all important libraries

# ### Load Data 

# In[2]:


hypothesis = pd.read_csv('/datasets/hypotheses_us.csv', sep = ';')
hypothesis.info()
pd.set_option('max_colwidth', 400)
display(hypothesis)


# Opened up data file and looked for the dataset and general information about dataset:
# 
# •	Hypotheses — brief descriptions of the hypotheses
# •	Reach — user reach, on a scale of one to ten
# •	Impact — impact on users, on a scale of one to ten
# •	Confidence — confidence in the hypothesis, on a scale of one to ten
# •	Effort — the resources required to test a hypothesis, on a scale of one to ten. The higher the Effort value, the more resource-intensive the test.
# 

# ### Apply the ICE framework to prioritize hypotheses 

# In[3]:



# for each row, apply ICE formula = (impact * confidence) / effort
hypothesis['ICE'] = hypothesis.apply(lambda x: (x['Impact'] * x['Confidence']) / x['Effort'], axis=1)

# sort in descending order
ice_results = hypothesis.sort_values(by='ICE', ascending=False)

display(ice_results)


# Each task enters one of the four quadrants: important and urgent, not important and urgent, important and not urgent, not important and not urgent. Hypotheses from the A quadrant (important and urgent) must be tested first. Then come hypotheses from the B quadrant (important and not urgent).
# For hypotheses, but not for tasks, those that are not important (the C and D quadrants) aren't tested at all. Urgent but unimportant hypotheses (the C quadrant) probably have short-term results that won't impact long-term business goals.
# 
# Impact, confidence, effort/ease (ICE) is one of the most popular ways of prioritizing problems:
# ICE Score = Impact x Confidence x Effort

# ### Apply the RICE framework to prioritize hypotheses 

# In[4]:


# for each row, apply RICE formula = (impact * confidence * reach) / effort
hypothesis['RICE'] = hypothesis.apply(lambda x: (x['Impact'] * x['Confidence'] * x['Reach']) / x['Effort'], axis=1)

# sort in descending order
rice_results = hypothesis.sort_values(by='RICE', ascending=False)

display(rice_results)


# RICE has four components:
# Reach — how many users will be affected by the update you want to introduce
# Impact — how strongly this update will affect the users, their experience, and their satisfaction with the product
# Confidence — how sure you are that your product will affect them in this way
# Effort — how much will it cost to test the hypothesis
# 
# RICE = (impact * confidence * reach) / effort

# ### Show how the prioritization of hypotheses changes when you use RICE instead of ICE 

# In[5]:


# plot the difference
ax = hypothesis[['Hypothesis','ICE','RICE']].plot(kind='bar',stacked=False, figsize=(8,6))

for p in ax.patches:
    ax.annotate(str(p.get_height().round()), (p.get_x() * 1.005, p.get_height() * 1.005), rotation=90)   
plt.title('ICE vs RICE Prioritization')
plt.xlabel('Hypotheses')
plt.ylabel('ICE/RICE Score')


# In[6]:


import plotly.express as px
df = px.data.tips()

df = hypothesis[['Hypothesis','ICE','RICE']]

fig = px.scatter(df, x="RICE", y="Hypothesis", text="ICE", log_x=True, size_max=400, color="ICE")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Hypothesis', title_x=0.5)
fig.show()
fig = px.scatter(df, x="ICE", y="Hypothesis", text="RICE", log_x=True, size_max=400, color="RICE")
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Hypothesis', title_x=0.5)
fig.show()



# The difference between ICE and RICE frameworks is the reach parameter, which estimates how many customers will be reached using polling. This parameters adds another factor to the framework, making RICE scores higher than ICE scores. As such, the RICE framework will be impacted by the reach factor and the RICE score will be skewed accordingly. For example, a hypothesis with a low impact confidence parameters will be measured differently with a high reach parameter. This can be seen in the data itself: in the ICE results, hypothesis 8 is the highest ranking with impact confidence being 81. However, in the RICE results hypothesis 8 falls to the fifth highest ranking because its reach parameter is low while, hypothesis 7 (which was third highest ranking in ICE results) climbs to the highest ranking becuase its reach parameter is the highest possible value.

# ## part 2:  Prioritizing Hypotheses

# In[7]:


orders = pd.read_csv('/datasets/orders_us.csv', sep=',')
orders['date'] = pd.to_datetime(orders['date'])


orders.info()
display(orders.head())


visits = pd.read_csv('/datasets/visits_us.csv', sep=',')
visits['date'] = pd.to_datetime(visits['date'])


visits.info()
display(visits.head())


# Here we have loaded data of Visits and orders which contains
# orders:
# 
# *   transactionId — order identifier
# *	visitorId — identifier of the user who placed the order
# *	date — of the order
# *	revenue — from the order
# *	group — the A/B test group that the user belongs to
# 
# Visits:
# 
# *	date — date
# *	group — A/B test group
# *	visits — the number of visits on the date specified in the A/B test group specified

# ### Data Preprocessing 

# In[8]:


orders.isnull().sum()


# In[9]:


visits.isnull().sum()


# In[10]:


orders = orders.dropna()
visits = visits.dropna()


# In[11]:


visits.drop_duplicates()


# In[12]:


visits_duplicate = visits[visits.duplicated()]
visits_duplicate


# In[13]:


orders.groupby('group').filter(lambda x: x.visitorId.is_unique)


# In[14]:


orders.groupby(['visitorId'])['group'].nunique()


# In[15]:


orders.groupby('visitorId', as_index=False).agg({'group':'nunique'}).query('group > 1')['visitorId']


# In[16]:


orders = orders[~orders['visitorId'].isin(orders.groupby('visitorId').filter(lambda group: (group.nunique() > 1).any())['visitorId'])]
orders


# Here we have checked for missing values and dropped duplicates.

# There no duplicates here

# ### Graph cumulative revenue by group. 

# In[17]:


dateGroups = orders[['date','group']].copy()


# In[18]:


dateGroups = dateGroups.drop_duplicates()


# Here we created new dataframe with date and group from Orders table to calculate unique paired group and date.

# In[19]:


ordersAgg = dateGroups.apply(
    lambda x: orders[np.logical_and(orders['date'] <= x['date'], orders['group'] == x['group'])]
    .agg({'date' : 'max', 'group' : 'max', 'transactionId' : pd.Series.nunique, 'visitorId' : pd.Series.nunique, 
          'revenue' : 'sum'}), axis=1).sort_values(by=['date','group'])


# DF with unique paired date and group values from orders DF

# In[20]:


visitsAgg = dateGroups.apply(
    lambda x: visits[np.logical_and(visits['date'] <= x['date'], visits['group'] == x['group'])]
    .agg({'date' : 'max', 'group' : 'max', 'visits' : 'sum'}), axis=1).sort_values(by=['date','group'])


# DF with unique paired date and group values from visits DF

# In[21]:


cumulativeData = ordersAgg.merge(visitsAgg, left_on=['date', 'group'],  right_on=['date','group'])
cumulativeData.columns = ['date','group','orders','buyers','revenue','visitors']
display(cumulativeData)


# merge two tables into one

# In[22]:


cumRevenueA = cumulativeData[cumulativeData['group'] == 'A'][['date','revenue','orders']]
cumRevenueB = cumulativeData[cumulativeData['group'] == 'B'][['date','revenue','orders']]


# cumulative orders and cumulative revenue by day for groups A/B

# In[23]:


plt.plot(cumRevenueA['date'], cumRevenueA['revenue'], label='A');
plt.plot(cumRevenueB['date'], cumRevenueB['revenue'], label='B');
plt.legend();
plt.title('Cumulative Revenue by Group over Time');
plt.xlabel('Time');
plt.ylabel('Cumulative Revenue');
plt.xticks(rotation=90);


# The cumulative revenue was calculated for both groups A and B, as shown in the graph above. As one can see, the cumulative revenue for both groups increases steadily overtime. However, around August 17 2019, the cumulative revenue of group B had a steap and swift increase from 40,000 dollars to 60,000 dollars and from there kept increasing upward past 80,000 dollars. While the cumulative revenue of group A did keep increasing, it did not see a sudden increase in cumulative revenue in August 17 2019.

# ### Graph cumulative average order size by group 

# In[24]:


plt.plot(cumRevenueA['date'], cumRevenueA['revenue']/cumRevenueA['orders'], label='A');
plt.plot(cumRevenueB['date'], cumRevenueB['revenue']/cumRevenueB['orders'], label='B');

plt.axhline(y=100, color='black', linestyle='--');
plt.legend();
plt.title('Average Purchase Size by Group over Time');
plt.xlabel('Time');
plt.ylabel('Average Number of Orders');
plt.xticks(rotation=90);


# diving the revenue by the cumulative number of orders.
# The cumulative average order size was calculated for both groups A and B, as shown in the graph above. Previously, we saw that group B has a sudden spike in cumulative revenue around August 17 2019 while group A did not have such increase. From this graph, we can see that the average purchase size was what caused a jump in cumulative revenue for group B. Although average purchase size for both groups is steadily increasing for the month of August 2019, group B sees a sudden jump from around 105 average orders to 160 average orders. After August 17 2019, the average number of orders decreased slowly but this did not affect the cumulative revenue.

# ###  Graph the relative difference in cumulative average order size for group B compared with group A

# In[25]:


mergedCumRevenue = cumRevenueA.merge(cumRevenueB, left_on='date', right_on='date', how='left', suffixes=['A','B'])


mergedCumRevenue['AverageA'] = mergedCumRevenue['revenueA'] / mergedCumRevenue['ordersA']
mergedCumRevenue['AverageB'] = mergedCumRevenue['revenueB'] / mergedCumRevenue['ordersB']

plt.plot(mergedCumRevenue['date'], mergedCumRevenue['AverageB'] - mergedCumRevenue['AverageA'], label='B-A');

plt.axhline(y=0, color='black', linestyle='--');
plt.legend();
plt.title('Relative Difference in Cumulative Average Order Size per Groups');
plt.xlabel('Time');
plt.ylabel('Average Order Size');
plt.xticks(rotation=90);
print(mergedCumRevenue.head())


# gather data into one dataframe with group suffixes.
# 
# Graph the relative difference in cumulative average order size for group B compared with group A

# ### Calculate each group's conversion rate as the ratio of orders to the number of visits for each day 

# In[26]:


cumulativeData['conversion'] = cumulativeData['orders'] / cumulativeData['visitors']
cumulativeData


# calculate and store conversion rate

# In[27]:


cumDataA = cumulativeData[cumulativeData['group'] == 'A']
cumDataB= cumulativeData[cumulativeData['group'] == 'B']


# select group data

# In[28]:


mergedCumConversion = cumDataA[['date','conversion','visitors']].merge(cumDataB[['date','conversion','visitors']], left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

Merged the data
# In[29]:


# plt.plot(mergedCumConversion['date'], mergedCumConversion['visitorsA']/mergedCumConversion['conversionA'], label='A');
# plt.plot(mergedCumConversion['date'], mergedCumConversion['visitorsB']/mergedCumConversion['conversionB'], label='B');
# plt.legend();
# plt.title('Conversion Rate');
# plt.xlabel('Time');
# plt.ylabel('Relativen Gain');
# plt.xticks(rotation=90);


# In[30]:


plt.plot(mergedCumConversion['date'], mergedCumConversion['conversionA'], label='A');
plt.plot(mergedCumConversion['date'], mergedCumConversion['conversionB'], label='B');
plt.legend();
plt.title('Conversion Rate');
plt.xlabel('Time');
plt.ylabel('Relativen Gain');
plt.xticks(rotation=90);


# This graph shows the conversion rates as the ratio of orders to the number of visits each day for each group. As the graph shows, both groups have aslight symmitrical conversion rate over time although group A has a greater relative gain than group B.

# ### Plot a scatter chart of the number of orders per user 

# In[31]:


order_users = orders.copy()


# In[ ]:


order_users= order_users.drop(['group','revenue','date'], axis=1).groupby('visitorId', as_index=False).agg({'transactionId':pd.Series.nunique})

order_users.columns = ['userId', 'orders']


# drop unneccessary columns and group the orders by users

# In[ ]:


order_users = order_users.sort_values(by='orders', ascending=False)


# sort data by number of orders in descending orders

# In[ ]:


x_values = pd.Series(range(0, len(order_users)));

plt.scatter(x_values, order_users['orders']);
plt.title('Raw Data: Number of Orders per User');
plt.xlabel('Number of Generated Observations');
plt.ylabel('Number of Orders');


# find values for horizontal axis by the number of generated observations.
# This scatter plot shows the number of orders per users using the raw data with anomalies. As one can see, the number of orders per user is mostly 1-2 orders. Very few users order above 3 or 4 units.

# ### Calculate the 95th and 99th percentiles for the number of orders per user 

# In[ ]:


print(np.percentile(order_users['orders'], [95,99]))


# With percentiles, we can determine the value below which n percent of observations fall. As we can see, 95% of the observations are within 2 orders per user and 99% of obercations are within 4 orders per user. We can define the point at which the data value becomes an anomaly as any observation where the number of orders is greater than 4 orders.

# ### Plot a scatter chart of order prices. 

# In[ ]:


order_price = orders.copy()


# In[ ]:


order_price= order_price.drop(['group','transactionId','date'], axis=1).groupby('visitorId', as_index=False).agg({'revenue':'sum'})
order_price.columns = ['userId', 'revenue']


# drop unneccessary columns and group the orders by users

# In[ ]:


order_price = order_price.sort_values(by='revenue', ascending=False)


# sort data by number of orders in descending orders

# In[ ]:


x_values = pd.Series(range(0, len(order_price)));

plt.scatter(x_values, order_price['revenue']);

plt.title('Raw Data: Revenue per Order');
plt.xlabel('Number of Generated Observations');
plt.ylabel('Revenue per Order');


# This scatter plot shows the revenue per order using raw data with possible anomalies. As one can see, the revenue per order is mostly before 2500 dollars. Very few orders have revenue above 2500 dollars.

# ### Calculate the 95th and 99th percentiles of order prices 

# In[ ]:


print(np.percentile(order_price['revenue'], [95,99]))


# With percentiles, we can determine the value below which n percent of observations fall. As we can see, 95% of the observations are within 510 dollars per order and 99% of obercations are within 1000 dollars per order. We can define the point at which the data value becomes an anomaly as any observation where the revenue is greater than 500 dollars.

# ### Find the statistical significance of the difference in conversion between the groups using the raw data 

# In[ ]:


order_usersA = orders[orders['group'] == 'A'].groupby('visitorId', as_index=False).agg({'transactionId': pd.Series.nunique})
order_usersA.columns = ['userId', 'orders']


# calculate statistical significance of difference in conversion between groups 

# In[ ]:


order_usersB = orders[orders['group'] == 'B'].groupby('visitorId', as_index=False).agg({'transactionId': pd.Series.nunique})
order_usersB.columns = ['userId', 'orders']


# Null Hypothesis H0: There is no statistically significant difference in conversion between groups A and B. Alternative Hypothesis H1: There is a statistically significant difference in conversion between groups A and B.
# 
# The p_value of 0.01 is less than the alpha level of 0.05 which means we can reject the null hypothesis and determine the difference between the conversion rate of groups A and B is statistically significant. This means there is a non-typical shift in the data.

# In[ ]:


tempA = pd.concat([order_usersA['orders'], pd.Series(0, index = np.arange(visits[visits['group'] == 'A']['visits'].sum() - len(order_usersA['orders'])), name='orders')], axis=0)


# delcare vars with users from different groups and the number of users / group

# In[ ]:


tempB = pd.concat([order_usersB['orders'], pd.Series(0, index = np.arange(visits[visits['group'] == 'B']['visits'].sum() - len(order_usersA['orders'])), name='orders')], axis=0)


# In[ ]:


p_value = st.mannwhitneyu(tempA, tempB)[1]
print(p_value)

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')


# We want to test the statistical significance of the difference in conversion between groups A and B. This can be done using the Mann-Wilcoxon-Whitney non-parametric test which ranks two samples in ascending order and compares the ranks of the values that appears in both samples. If the differences between their ranks are the same from sample to sample, it means the shirt is typical (some values were added, causing the rest of the values to shift). On the other hand, a non-typical shift means a real change occurred and the sum of such shifts in rank if the value of the criterion
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between groups A and B. Alternative Hypothesis H1: There is a statistically significant difference in conversion between groups A and B.
# 
# The p_value of 0.01 is less than the alpha level of 0.05 which means we can reject the null hypothesis and determine the difference between the conversion rate of groups A and B is statistically significant. This means there is a non-typical shift in the data.

# ### Find the statistical significance of the difference in average order size between the groups using the raw data 

# In[ ]:


p_value = st.mannwhitneyu(orders[orders['group'] == 'A']['revenue'], orders[orders['group'] == 'B']['revenue'])[1]
print("{0:.3f}".format(p_value))
alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')


# We want to test the statistical significance of the difference in average order size between groups A and B. This can be done using the Mann-Wilcoxon-Whitney non-parametric test which ranks two samples in ascending order and compares the ranks of the values that appears in both samples. If the differences between their ranks are the same from sample to sample, it means the shift is typical (some values were added, causing the rest of the values to shift). On the other hand, a non-typical shift means a real change occurred and the sum of such shifts in rank if the value of the criterion
# 
# Null Hypothesis H0: There is no statistically significant difference in average order size between groups A and B. Alternative Hypothesis H1: There is a statistically significant difference in average order size between groups A and B.
# 
# The p_value of 0.35 is greater than than the alpha level of 0.05 which means we can cannnot reject to the null hypothesis and cannot make any conclusions about the difference in average order size. Additionally, it means there is a typical shift in the data.

# ### Find the statistical significance of the difference in conversion between the groups using the filtered data. 

# In[ ]:


user_more_order = pd.concat([order_usersA[order_usersA['orders'] > 2]['userId'], order_usersB[order_usersB['orders'] > 2]['userId']], axis = 0)


# identify anomalous users with tooo many orders

# In[ ]:


user_costly_order = orders[orders['revenue'] > 500]['visitorId']


# identify anomalous users with expensive orders

# In[ ]:


abnormal_user = pd.concat([user_more_order, user_costly_order], axis = 0).drop_duplicates().sort_values()


# join them into abnormal table and remove dupliated

# In[ ]:


tempA_filter = pd.concat([order_usersA[np.logical_not(order_usersA['userId'].isin(abnormal_user))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='A']['visits'].sum() - len(order_usersA['orders'])),name='orders')],axis=0)


# calculate statistical signifiacne of the difference in conversion between groups using filtered data

# In[ ]:


tempB_filter = pd.concat([order_usersB[np.logical_not(order_usersB['userId'].isin(abnormal_user))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='B']['visits'].sum() - len(order_usersB['orders'])),name='orders')],axis=0)


# In[ ]:


p_value = st.mannwhitneyu(tempA_filter, tempB_filter)[1]
print("{0:.5f}".format(p_value))

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')


# The raw data was filtered by two criterion, the number of orders and the revenue per order. As previously determined, 95% of users make 2 orders and the average revenue per order is about 500 dollars. We can filter the data such that we only have rows with 2 or less orders and that have a revenue of 500 dollars of less.
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between groups A and B. Alternative Hypothesis H1: There is a statistically significant difference in conversion between groups A and B.
# 
# The p_value of 0.005 is still less than the alpha level of 0.05 which means we can reject the null hypothesis and determine the difference between the conversion rate of groups A and B is statistically significant. This means there is a non-typical shift in the data.

# ### Find the statistical significance of the difference in average order size between the groups using the filtered data 

# In[ ]:


p_value = st.mannwhitneyu(
    orders[np.logical_and(
        orders['group']=='A',
        np.logical_not(orders['visitorId'].isin(abnormal_user)))]['revenue'],
    orders[np.logical_and(
        orders['group']=='B',
        np.logical_not(orders['visitorId'].isin(abnormal_user)))]['revenue'])[1]
print("{0:.3f}".format(p_value))

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')


# The raw data was filtered by two criterion, the number of orders and the revenue per order. As previously determined, 95% of users make 2 orders and the average revenue per order is about 500 dollars. We can filter the data such that we only have rows with 2 or less orders and that have a revenue of 500 dollars of less.
# 
# Null Hypothesis H0: There is no statistically significant difference in average order size between groups A and B. Alternative Hypothesis H1: There is a statistically significant difference in average order size between groups A and B.
# 
# The p_value of 0.35 is greater than than the alpha level of 0.05 which means we can cannnot reject to the null hypothesis and cannot make any conclusions about the difference in average order size. Additionally, it means there is a typical shift in the data.

# ## Make a decision based on the test results 

# From the above calculations and graphs, it is safe to make a data-driven recommendation to marketing specialists that group B is the better group in which in invest resources. They generate greater cumulative revenue and have higher average order sizes.
# 
# We can stop the test and go with group B

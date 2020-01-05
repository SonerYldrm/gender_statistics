#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Specify style of seaborn visualizations
sns.set(style="darkgrid")


# # Importing and Cleaning Data

# In[4]:


data = pd.read_csv('data.csv')


# In[5]:


data.shape


# In[6]:


data.head()


# There are more than 500 indicators. My plan for this study is to use 25 indicators to compare 4 different countries in different continents. Each indicator has a code so I will create list of the indicators that will be used. Then, I will filter out the dataframe to include only these indicators.

# In[7]:


code_list = ['SL.FAM.WORK.FE.ZS','SL.FAM.WORK.MA.ZS','IC.REG.COST.PC.FE.ZS','IC.REG.COST.PC.MA.ZS',
             'SP.DYN.CDRT.IN','SL.AGR.EMPL.FE.ZS','SL.AGR.EMPL.MA.ZS','SL.IND.EMPL.FE.ZS',
             'SL.IND.EMPL.MA.ZS','SL.SRV.EMPL.FE.ZS','SL.SRV.EMPL.MA.ZS','SE.TER.GRAD.FE.SI.ZS',
             'SP.DYN.TFRT.IN','SL.TLF.ACTI.1524.FE.NE.ZS','SL.TLF.ACTI.1524.MA.NE.ZS',
             'SL.TLF.CACT.FE.NE.ZS','SL.TLF.CACT.MA.NE.ZS','SL.TLF.TOTL.FE.ZS','SP.DYN.LE00.FE.IN',
             'SP.DYN.LE00.MA.IN','SP.POP.0014.TO.ZS','SP.POP.1564.TO.ZS','SP.POP.65UP.TO.ZS',
             'SH.STA.OB18.FE.ZS','SH.STA.OB18.MA.ZS'
]


# In[8]:


data.shape


# In[9]:


mask = data['Series Code'].isin(code_list)
data1 = data[mask]


# In[10]:


data1.shape


# We now have 25 attributes of 4 different countries for 15 years (2004-2018).

# The column names can be simplified. For example the year columns include years twice.

# In[12]:


data1.columns = ['series_name', 'series_code', 'country_name', 'country_code', '2004', '2005', '2006', '2007', 
                '2008', '2009', '2010', '2011','2012', '2013', '2014', '2015', '2016', '2017', '2018']


# In[13]:


#country name column can be dropped since country code clearly tells us the country
data1 = data1.drop('country_name', axis=1)


# In[323]:


# ALways check your work 
data1.head()


# ### Missing values

# Handling missing values is a critical step in data cleaning. We first need to find the missing values and then decide how to handle them depending on the characteristics and properties of the data. There is not an optimal way to do this for all datasets. It will largely depend on the dataset.

# In[236]:


data1.isna().any().sum()


# Although the count of missing values in our dataset is zero, I know there are missing values in the dataset. There are some values with '..' values which are actually missing values but could not be detected by pandas as missing values. Therefore, it is very important to also visually check the datasets. I will replace these values with NaN.

# In[19]:


data1.replace('..', np.nan, inplace=True)


# In[21]:


data1.isna().sum()


# 2018 columns contains 40 NaN out of 100 total values so I will drop the whole column.

# In[22]:


data1 = data1.drop('2018', axis=1)


# As expected, missing values are mostly in the same rows. Certain attributes do not have any values for a particular country which causes most of the values in a row to be NaN. So I will drop rows based on a threshold of NaN values in that row. 

# In[23]:


# 14 years (2004-2017) with 70 % threshold
data1 = data1.dropna(thresh=14*0.7, axis='rows')


# In[24]:


data1.isna().sum()


# In[25]:


data1.shape


# There are 14 missing values left. It is acceptible to fill these values with the value of a previous year of the same country which will not effect the result very much. To do this, I will use ffill method on rows.

# In[26]:


data1 = data1.fillna(method='ffill', axis=0)


# In[27]:


data1.isna().sum().sum()


# In[28]:


data1.columns


# The datatypes of the values are object which is not appropriate for numerical analysis. Therefore, I need to change the datatypes of numerical columns.

# In[29]:


years = ['2004', '2005', '2006',
       '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
       '2016', '2017']
for year in years:
    data1 = data1.astype({year:'float'})


# In[30]:


data1.dtypes


# In[31]:


data1.columns


# In[34]:


data1.head()


# In[36]:


data1.series_name.value_counts()


# My analysis aims to compare 4 different countries based on 25 different attributes for a period of 15 years. Due to unavailable data, I have dropped some columns and rows. After dropping these rows and columns, some attributes do not have data for all 4 countries. In order to have a complete and thorough analysis, I will only use attributes which have data for all 4 countries.

# In[37]:


attributes = data1.series_name.value_counts() == 4


# In[38]:


# Filter attributes with 4 values 
attributes = attributes[attributes.values == True]


# In[39]:


attributes


# In[40]:


attributes_list = list(attributes.index)


# In[41]:


attributes_list


# In[42]:


mask1 = data1['series_name'].isin(attributes_list)
data1 = data1[mask1]


# In[43]:


data1.shape


# We now have 18 attributes for 4 different countries for a period of 14 years.

# # Exploratory Data Analysis

# ## Employment

# The employment attributes indicate the labor force occupation of males and females. Labor force is divided into three occupations: agriculture, service and industry. I will filter these attributes using the description in series_name column.

# In[44]:


employment = data1[data1.series_name.str.contains('Employment')]


# In[45]:


employment.shape


# In[234]:


employment.head()


# 6 attributes for 4 countries:
# Employment in (agriculture, service, industry) for (male, female)

# I will try to make series codes more informative and drop series_name columns so that it will be easier to analysize and make visualizations. For ex, SL.AGR.EMPL.FE.ZS means employment in agriculture (female). I will export AGR and FE from this code and make separate columns. 

# In[47]:


split1 = employment.series_code.str.split(".", expand=True)


# In[48]:


split1.shape


# In[49]:


employment = pd.concat([employment, split1], axis=1)


# In[50]:


employment.head()


# In[51]:


employment = employment.drop(['series_code', 0, 2, 4], axis=1)


# In[52]:


employment.head()


# In[53]:


employment.rename(columns={1:'field', 3:'gender'}, inplace=True)


# In[54]:


employment.columns


# In[55]:


employment_new = employment.drop('series_name', axis=1)


# In[56]:


employment_new.head()


# Years listed as columns are not convenient for analysis and making visualizations. Therefore, I will create a year column and list all years as a row item. Pandas has built in melt function to do this task.

# In[57]:


employment_new = employment_new.melt(id_vars=['country_code', 'field', 'gender'],
                   var_name='year',
                   value_name='employment')


# In[58]:


print(employment_new.shape)
employment_new.head()


# ### Occupations

# Let's check how the share of each occupation changed overtime for both males and females.

# In[59]:


# Create masks to filter employment dataframe
mask00 = (employment_new.field == 'AGR') & (employment_new.gender == 'MA')
mask01 = (employment_new.field == 'AGR') & (employment_new.gender == 'FE')
mask10 = (employment_new.field == 'SRV') & (employment_new.gender == 'MA')
mask11 = (employment_new.field == 'SRV') & (employment_new.gender == 'FE')
mask20 = (employment_new.field == 'IND') & (employment_new.gender == 'MA')
mask21 = (employment_new.field == 'IND') & (employment_new.gender == 'FE')


# In[284]:


fig, axs = plt.subplots(ncols=2, figsize=(18,6))
sns.set_context("notebook", font_scale=1.3, rc={"lines.linewidth": 2.})
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask00], legend=False, ax=axs[0])
axs[0].set_title('Male Employment in Agriculture')
axs[0].set_ylabel('% Share')
axs[0].set_xlabel('')
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask01], ax=axs[1])
axs[1].set_title('Female Employment in Agriculture')
axs[1].set_ylabel('')
axs[1].set_xlabel('')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[329]:


fig, axs = plt.subplots(ncols=2, figsize=(20,6))
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask10], legend=False, ax=axs[0])
axs[0].set_title('Male Employment in Service')
axs[0].set_ylabel('% Share')
axs[0].set_xlabel('')
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask11], ax=axs[1])
axs[1].set_title('Female Employment in Service')
axs[1].set_ylabel('')
axs[1].set_xlabel('')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[328]:


fig, axs = plt.subplots(ncols=2, figsize=(20,6))
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask20], legend=False, ax=axs[0])
axs[0].set_title('Male Employment in Industry')
axs[0].set_ylabel('% Share')
axs[0].set_xlabel('')
sns.lineplot(x='year', y='employment', hue='country_code', data=employment_new[mask21], ax=axs[1])
axs[1].set_title('Female Employment in Industry')
axs[1].set_ylabel('')
axs[1].set_xlabel('')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ### Countries

# The trends in each country may be realized more easily if each occupation and gender is shown on the same graph.

# In[272]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='employment', hue='field', style='gender', ci=None, data=employment_new)
ax.set_title('Labor Force Occupations in General')
ax.set_xlabel('')
ax.set_ylabel('% share')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[266]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
maskNLD = employment_new.country_code == 'NLD'
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='employment', hue='field', style='gender', data=employment_new[maskNLD])
ax.set_title('Employment in Netherlands')
ax.set_xlabel('')
ax.set_ylabel('% share')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[267]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
maskCHN = employment_new.country_code == 'CHN'
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='employment', hue='field', style='gender', data=employment_new[maskCHN])
ax.set_title('Employment in China')
ax.set_xlabel('')
ax.set_ylabel('% share')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[268]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
maskUSA = employment_new.country_code == 'USA'
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='employment', hue='field', style='gender', data=employment_new[maskUSA])
ax.set_title('Employment in United States')
ax.set_xlabel('')
ax.set_ylabel('% share')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[269]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
maskTUR = employment_new.country_code == 'TUR'
plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='employment', hue='field', style='gender', data=employment_new[maskTUR])
ax.set_title('Employment in Turkey')
ax.set_xlabel('')
ax.set_ylabel('% share')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In Turkey, the distribution of male employment types did not change a lot between 2004 and 2017. However, female jobs shifted from agriculture to service. Women in industry are a lot lower than women in service and agriculture.

# ## Obesity

# I will filter attributes related to obesity using the description in series_name column.

# In[66]:


obesity = data1[data1.series_name.str.contains('obesity')]


# In[67]:


obesity


# In[68]:


obesity.series_code[0:4] = 'FE'
obesity.series_code[4:9] = 'MA'


# In[69]:


obesity


# In[70]:


obesity = obesity.drop('series_name', axis=1)


# In[71]:


obesity = obesity.drop('2017', axis=1)


# In[72]:


obesity = obesity.rename(columns={'series_code':'gender'})


# In[73]:


obesity.head()


# In[74]:


obesity = obesity.melt(id_vars=['country_code', 'gender'],
                   var_name='year',
                   value_name='share')


# In[216]:


obesity.shape


# In[75]:


obesity.head()


# In[290]:


plt.figure(figsize=(12,8))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
ax = sns.lineplot(x='year', y='share', hue='country_code', style='gender', data=obesity)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Obesity in Different Countries')
ax.set_xlabel('')
ax.set_ylabel('% share')


# ## Age distribution, life expectancy, fertility rate and death rate

# Just like obesity and employment, I will filter attributes using the description in series_name column and combine different dataframes based on year and country columns using pandas merge() function.

# In[76]:


population = data1[data1.series_name.str.contains('Population')]
population.shape


# In[77]:


population.head()


# In[78]:


split2 = population.series_code.str.split(".", expand=True)


# In[79]:


population['ages'] = split2[2]


# In[80]:


population.head()


# In[81]:


population = population.drop(['series_name','series_code'], axis=1)


# In[82]:


population.head()


# In[83]:


population = population.melt(id_vars=['country_code', 'ages'],
                   var_name='year',
                   value_name='share')


# In[84]:


population.head()


# In[85]:


life_exp = data1[data1.series_name.str.contains('expectancy')]
life_exp.shape


# In[86]:


life_exp.head()


# In[87]:


split3 = life_exp.series_code.str.split('.', expand=True)


# In[88]:


split3


# In[89]:


life_exp['gender'] = split3[3]


# In[90]:


life_exp = life_exp.drop(['series_name', 'series_code'], axis=1)


# In[91]:


life_exp


# In[92]:


life_exp = life_exp.melt(id_vars=['country_code', 'gender'],
                        var_name='year',
                        value_name='expected_life' )


# In[93]:


life_exp.head()


# In[94]:


life_exp.shape


# In[96]:


population.head()


# In[97]:


population_new = pd.merge(population, life_exp, on=['country_code','year'])


# In[98]:


population_new.head(10)


# Death rate and fertility rate

# In[103]:


fertility = data1[data1.series_code == 'SP.DYN.TFRT.IN']


# In[104]:


fertility


# In[105]:


fertility = fertility.drop(['series_name','series_code'], axis=1)


# In[106]:


fertility = fertility.melt(id_vars='country_code',
                          var_name='year',
                          value_name='fertility_rate')


# In[107]:


fertility.head()


# In[108]:


population_new = pd.merge(population_new, fertility, on=['country_code','year'])


# In[109]:


population_new.head()


# In[110]:


death = data1[data1.series_code == 'SP.DYN.CDRT.IN']


# In[111]:


death.head()


# In[112]:


death = death.drop(['series_name','series_code'], axis=1)


# In[113]:


death = death.melt(id_vars='country_code',
                  var_name='year',
                  value_name='death_rate')


# In[114]:


death.head()


# In[115]:


population_new = pd.merge(population_new, death, on=['country_code','year'])


# In[156]:


population_new.head()


# Fertility rate: Total fertility rate represents the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with age-specific fertility rates of the specified year.

# Death rate: Crude death rate indicates the number of deaths occurring during the year, per 1,000 population estimated at midyear. Subtracting the crude death rate from the crude birth rate provides the rate of natural increase, which is equal to the rate of population change in the absence of migration.
# 

# In[292]:


plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='share', hue='ages', data=population_new)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Population Distribution')


# In[310]:


mask1 = population_new.ages == '0014'
mask2 = population_new.ages == '65UP'

fig, axs = plt.subplots(ncols=2, figsize=(20,6))

sns.lineplot(x='year', y='share', hue='country_code', data=population_new[mask1], legend=False, ax=axs[0])
axs[0].set_title('14 and under')
axs[0].set_ylabel('% Share')
axs[0].set_xlabel('')

sns.lineplot(x='year', y='share', hue='country_code', data=population_new[mask2], ax=axs[1])
axs[1].set_title('65 and UP')
axs[1].set_ylabel('')
axs[1].set_xlabel('')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[315]:


fig, axs = plt.subplots(ncols=2, figsize=(20,6))

sns.lineplot(x='year', y='fertility_rate', hue='country_code', data=population_new, legend=False, ax=axs[0])
axs[0].set_title('Fertility Rate')
axs[0].set_ylabel('Fertility')
axs[0].set_xlabel('')

sns.lineplot(x='year', y='expected_life', hue='country_code', ci=None, data=population_new, ax=axs[1])
axs[1].set_title('Life Expectancy')
axs[1].set_ylabel('Years')
axs[1].set_xlabel('')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[322]:


plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='expected_life', hue='country_code', style='gender',
             data=population_new)
ax.set_title('Gender Difference on Life Expectancy')
ax.set_ylabel('Years')
ax.set_xlabel('')

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# It is safe to say that women live longer than men in general. 

# In[332]:


plt.figure(figsize=(10,6))
ax = sns.lineplot(x='year', y='death_rate', hue='country_code',
             data=population_new)
ax.set_title('Death Rate')
ax.set_ylabel('Years')
ax.set_xlabel('')

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:





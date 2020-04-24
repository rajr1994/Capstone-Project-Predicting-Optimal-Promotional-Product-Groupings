# # Industry Practicum - Determining Effective Promotional Product Groupings

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'auto')
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import datetime as datetime
from getpass import getpass
import pymysql as mariadb
from sshtunnel import SSHTunnelForwarder
import pandas as pd
import random

# ## Uploading data from database
# Enter login details
username = getpass('username: ')
password = getpass('password: ')

# Open an ssh-tunnel via Scholar to Datamine
tunnel = SSHTunnelForwarder(('scholar.rcac.purdue.edu', 22), 
                            ssh_username=username, ssh_password=password,
                            remote_bind_address=('datamine.rcac.purdue.edu', 3306),
                            local_bind_address=('localhost', 3307))
tunnel.start()

# Connect to the database
connection = mariadb.connect(host=localhost, port=port, 
                             db=dbname, user=dbusername, password=dbpassword)

# Loading the dataset
raw_data = pd.read_sql('select * from raw_data', connection)
ppg_names = pd.read_sql('select * from Current_PPG_Names', connection)


# ## Data Cleaning and Preparation
# Rename sub brand1 as one unique value
c=0
for i in raw_data['UPC_Unit']:
    c += 1
    if i == 9300830045983:
        raw_data['Sub_Brand_EPOS'][c] = "Sub_Brand1"

# Excluding UPC Codes where Product Names and Size are not matching
data0 = pysqldf("SELECT * FROM raw_data WHERE UPC_Unit NOT IN (884486046680, 8850006493144, 9300701412456, 9310714223291);")

# Remove an erroneous particular record
data01 = sqldf("select * from data0 where EPOS_Product_Name <> 'Product2';")

# Aggregating the dataset
data1 = pysqldf("SELECT UPC_Unit, Customer, EPOS_Link_Date, Category_EPOS, Product_Category_EPOS, Sub_Category_EPOS, Manufacturer, Brand_EPOS, Sub_Brand_EPOS, Variant_EPOS, Pack_Type, Size_EPOS, Segment_EPOS, Size_Range_EPOS, Unit_of_Measure, Short_Segment, MAX(Store_Count_EPOS) as Store_Count_EPOS, SUM(Unit_Sales_EPOS) as Unit_Sales_EPOS, SUM(Value_Sales_EPOS) as Value_Sales_EPOS FROM data01 GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;")

# Removing duplicate rows in PPG file
ppg_data1 = pysqldf("SELECT UPC_Unit, Customer, EPOS_Link_Date, Example_PPg_Bundle, row_number() OVER (PARTITION BY UPC_Unit,Customer,EPOS_Link_Date order by Example_PPg_Bundle) as rank FROM ppg_names;")
ppg_data2 = pysqldf("SELECT UPC_Unit, Customer, EPOS_Link_Date, Example_PPg_Bundle FROM ppg_data1 where rank = 1;")

# Merging datasets - Raw Data with PPG Names
data2 = pysqldf("SELECT a.*, b.Example_PPg_Bundle FROM data1 a left join ppg_data2 b ON a.UPC_Unit = b.UPC_Unit AND a.EPOS_Link_Date = b.EPOS_Link_Date AND a.Customer = b.Customer;")

# Rounding off Unit Sales (Quantities) to 0
data2 = data2.round({"Unit_Sales_EPOS":0})

# Calculating Price from Sales Value and Quantities
data2['Price'] = data2['Value_Sales_EPOS']/ data2['Unit_Sales_EPOS']
data2 = data2.round({"Price":2})

# Creating a lookup table for UPC Code and Product Name
prod_lookup = pysqldf("SELECT UPC_Unit, EPOS_Product_Name, row_number() OVER (PARTITION BY UPC_Unit ORDER BY Value_Sales_EPOS desc) AS rank FROM data0;")
prod_lookup_final = pysqldf("SELECT UPC_Unit, EPOS_Product_Name FROM prod_lookup WHERE rank = 1;")

# Filtering negative values in Revenue and Quantities
data3 = data2[(data2['Unit_Sales_EPOS']>0) & (data2['Value_Sales_EPOS']>0)]


# ## Feature Engineering
# Calculating Maximum Price per UPC Code
max_price = pysqldf("SELECT UPC_Unit, MAX(Price) as max_price FROM data3 GROUP BY 1;")
data4 = pysqldf("SELECT a.*, max_price from data3 a inner join max_price b on a.UPC_Unit = b.UPC_Unit;")

# Create Promotion Column - '1' if the product is sold at price less than 94% of max_price
data4['Promotions_1_0'] = np.where(data4['Price'] < 0.94 * data4['max_price'], 1, 0)

# Create Discount Column - calculate discount for products with promotions
data4['Discount'] = np.where(data4['Promotions_1_0'] == 1, data4['max_price'] - data4['Price'], 0)

# Create percentage Discount Column
data4['p_discount'] = np.where(data4['Promotions_1_0'] == 1, data4['Discount']/data4['max_price'], 0)

#Converting date to YMD format
data4['EPOS_Link_Date'] = data4['EPOS_Link_Date'].astype(str)
data4['EPOS_Link_Date'] = data4['EPOS_Link_Date'].str.slice(0, -12)
data4['date'] = data4['EPOS_Link_Date'].replace(" ", "")
data4['date'] = pd.to_datetime(data4['date'])

# Find first_date at UPC_Unit and Customer level
group1 = pysqldf("select UPC_Unit, Customer, min(date) as min_date from data4 where Store_Count_EPOS >= 100 group by 1,2;")
group2a =  pysqldf("select a.UPC_Unit,a.Customer,Store_Count_EPOS, date from data4 a join group1 b on a.UPC_Unit = b.UPC_Unit and a.Customer = b.Customer where Store_Count_EPOS < 100 and date >= min_date;")
group2b = pysqldf("select UPC_Unit,Customer,date,Store_Count_EPOS, row_number() over(partition by UPC_Unit,Customer order by Store_Count_EPOS desc) as rn from group2a;")

#Find promotion end date
group2c = pysqldf("select UPC_Unit,Customer, date as promotion_end_date from group2b where rn = 1;")
duration = pysqldf("select a.UPC_Unit,a.Customer,promotion_end_date,min_date from group1 a left join group2c b on a.UPC_Unit = b.UPC_Unit and a.Customer = b.Customer;")

#Cleaning promotion end date column (replace nulls and convert to datetime)
duration['promotion_end_date'].fillna(max(data4['date']), inplace = True)
duration['promotion_end_date'] = pd.to_datetime(duration['promotion_end_date'])
duration['min_date'] = pd.to_datetime(duration['min_date'])

#Finding duration of promotional period - calculated from start of promotion to date when store count of the product falls below 100
duration['duration_days'] = duration['promotion_end_date'].sub(duration['min_date'], axis=0)
duration['duration_days'] = duration['duration_days'] / np.timedelta64(1, 'D')
duration['duration_days'] = duration['duration_days'] + 7


data5 = pysqldf("select a.*,coalesce(duration_days,0) as duration_days from data4 a left join duration b on a.UPC_Unit = b.UPC_Unit and a.Customer = b.Customer;")


# ## Determining new PPGs (for Modeling)
#Standardizing Overall Sales, Units Sold and Average Days of Promotions to generate score for each PPG
det1 = pysqldf("select Sub_Category_EPOS, Size_Range_EPOS, Example_PPg_Bundle, sum(Value_Sales_EPOS) as Value_Sales_EPOS, sum(Unit_Sales_EPOS) as Unit_Sales_EPOS, avg(duration_days) as avg_duration_days from data5 group by 1,2,3;")
x = det1[['Value_Sales_EPOS']].values.astype(float)
y = det1[['Unit_Sales_EPOS']].values.astype(float)
z = det1[['avg_duration_days']].values.astype(float)

#Standardizing using Min-Max Scaler
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
y_scaled = min_max_scaler.fit_transform(y)
z_scaled = min_max_scaler.fit_transform(z)
det1_normalized = pd.DataFrame(x_scaled)
det1_normalized2 = pd.DataFrame(y_scaled)
det1_normalized3 = pd.DataFrame(z_scaled)

#Replace the three columns with standardized values (between 0 and 1)
det1['Value_Sales_EPOS'] = det1_normalized
det1['Unit_Sales_EPOS'] = det1_normalized2
det1['avg_duration_days'] = det1_normalized3

# Calculate a score with revenue,quantity and duration
det1['score'] = (det1['Value_Sales_EPOS'] + det1['Unit_Sales_EPOS'] + det1['avg_duration_days'])/3

#Removing scientific notation
det1['Value_Sales_EPOS'] = det1['Value_Sales_EPOS'].apply(lambda x: '{:.8f}'.format(x))
det1['Unit_Sales_EPOS'] = det1['Unit_Sales_EPOS'].apply(lambda x: '{:.8f}'.format(x))
det1['avg_duration_days'] = det1['avg_duration_days'].apply(lambda x: '{:.8f}'.format(x))
det1['score'] = det1['score'].apply(lambda x: '{:.8f}'.format(x))

# Determine the PPG bundle having the max score
det2 = pysqldf("select a.Sub_Category_EPOS,Size_Range_EPOS,Example_PPg_Bundle, row_number() over(partition by Sub_Category_EPOS,Size_Range_EPOS order by score desc) as rn from det1 a;")
det3 = pysqldf("select a.* from det2 a where rn = 1;")
data6 = pysqldf("select a.*,b.Example_PPg_Bundle as model_PPg from data5 a left join det3 b on a.Sub_Category_EPOS = b.Sub_Category_EPOS and a.Size_Range_EPOS = b.Size_Range_EPOS;")
data6['y'] = np.where(data6['Example_PPg_Bundle'] == data6['model_PPg'], 1, 0)


# ## Modeling for Individual Subcategories - Sub-Category1
#Creating the subset
sub_cat1 = data6.loc[(data6['Manufacturer'] == "Maunfacturer1") & (data6['Sub_Category_EPOS'] == "Sub_Category1")]

# Subset X Predictors and Y
sub_cat1 = sub_cat1[['y', 'Variant_EPOS', 'Pack_Type', 'Segment_EPOS', 'Size_Range_EPOS', 'Short_Segment', 'Promotions_1_0']]

#Creating dummy variables
sub_cat1 = pd.get_dummies(sub_cat1,drop_first=True)

#Checking whether y is balanced or not
target_count = sub_cat1.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#Zeroes are undersampled; imbalanced data


# Random Oversampling of Zeroes
count_class_0, count_class_1 = sub_cat1.y.value_counts()

# Divide by class
sub_cat1_class_0 = sub_cat1[sub_cat1['y'] == 0]
sub_cat1_class_1 = sub_cat1[sub_cat1['y'] == 1]

sub_cat1_class_0_over = sub_cat1_class_0.sample(count_class_0, replace=True,random_state=0)
sub_cat1_test_over = pd.concat([sub_cat1_class_1, sub_cat1_class_0_over], axis=0)

print('Random over-sampling:')
print(sub_cat1_test_over.y.value_counts())

sub_cat1 = sub_cat1_test_over

#Logistic regression
import statsmodels.api as sm
X = sub_cat1.loc[:, sub_cat1.columns != 'y']
y = sub_cat1[['y']].copy()

logitm=sm.Logit(y.astype('float'),X.astype('float'))
result=logitm.fit(method='bfgs')
print(result.summary())

#Generating the confusion matrix
pred = result.pred_table(threshold=.5)
print(pred)

#Generating accuracy
acc = round((pred[0][0] + pred[1][1])/(pred[0][0] + pred[1][1] + pred[1][0] + pred[0][1])*100, 2)
print('Accuracy = ', acc, '%')


# ## Modeling for Individual Subcategories - Sub-Category2
#Creating the subset
sub_cat2 = data6.loc[(data6['Manufacturer'] == "Manufacturer2") & (data6['Sub_Category_EPOS'] == "Sub_Category2")]

# Subset X Predictors and Y
sub_cat2 = sub_cat2[['y', 'Variant_EPOS', 'Pack_Type', 'Segment_EPOS', 'Size_Range_EPOS', 'Short_Segment', 'Promotions_1_0']]

#Creating dummy variables
sub_cat2 = pd.get_dummies(sub_cat2,drop_first=True)

#Checking whether y is balanced or not
target_count = sub_cat2.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#Ones are undersampled; imbalanced data


#Random Oversampling of Ones
count_class_0, count_class_1 = sub_cat2.y.value_counts()

# Divide by class
sub_cat2_class_0 = sub_cat2[sub_cat2['y'] == 0]
sub_cat2_class_1 = sub_cat2[sub_cat2['y'] == 1]


sub_cat2_class_1_over = sub_cat2_class_1.sample(count_class_0, replace=True,random_state=0)
sub_cat2_test_over = pd.concat([sub_cat2_class_0, sub_cat2_class_1_over], axis=0)

print('Random over-sampling:')
print(sub_cat2_test_over.y.value_counts())

sub_cat2 = sub_cat2_test_over

#Logistic regression
import statsmodels.api as sm
X = sub_cat2.loc[:, sub_cat2.columns != 'y']
y = sub_cat2[['y']].copy()

logitm=sm.Logit(y.astype('float'),X.astype('float'))
result=logitm.fit(method='bfgs')
print(result.summary())

#Generating the confusion matrix
pred = result.pred_table(threshold=.5)
print(pred)

#Generating accuracy
acc = round((pred[0][0] + pred[1][1])/(pred[0][0] + pred[1][1] + pred[1][0] + pred[0][1])*100, 2)
print('Accuracy = ', acc, '%')


# ## Modeling for Individual Subcategories - Sub-Category3
#Creating the subset
sub_cat3 = data6.loc[(data6['Manufacturer'] == "Manufacturer3") & (data6['Sub_Category_EPOS'] == "Sub_Category3")]

# Subset Y and X Predictors
sub_cat3 = sub_cat3[['y', 'Variant_EPOS', 'Pack_Type', 'Segment_EPOS', 'Size_Range_EPOS', 'Short_Segment', 'Promotions_1_0']]

#Creating dummy variables
sub_cat3 = pd.get_dummies(sub_cat3,drop_first=True)

#Checking whether y is balanced or not
target_count = sub_cat3.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

#Zeroes are undersampled; imbalanced data

# Random Oversampling of Zeroes

count_class_0, count_class_1 = sub_cat3.y.value_counts()

# Divide by class
sub_cat3_class_0 = sub_cat3[sub_cat3['y'] == 0]
sub_cat3_class_1 = sub_cat3[sub_cat3['y'] == 1]

sub_cat3_class_0_over = sub_cat3_class_0.sample(count_class_0, replace=True,random_state=0)
sub_cat3_test_over = pd.concat([sub_cat3_class_1, sub_cat3_class_0_over], axis=0)

print('Random over-sampling:')
print(sub_cat3_test_over.y.value_counts())

sub_cat3 = sub_cat3_test_over

#Logistic regression
import statsmodels.api as sm
X = sub_cat3.loc[:, sub_cat3.columns != 'y']
y = sub_cat3[['y']].copy()

logitm=sm.Logit(y.astype('float'),X.astype('float'))
result=logitm.fit(method='bfgs')
print(result.summary())

#Generating the confusion matrix
pred = result.pred_table(threshold=.5)
print(pred)

#Generating accuracy
acc = round((pred[0][0] + pred[1][1])/(pred[0][0] + pred[1][1] + pred[1][0] + pred[0][1])*100, 2)
print('Accuracy = ', acc, '%')

# ## Modeling for Individual Subcategories - Sub-Category4
#Creating the subset
random.seed(123)
sub_cat4 = data6.loc[(data6['Manufacturer'] == "Manufacturer4") & (data6['Sub_Category_EPOS'] == "Sub_Category4")]

# Subset Y and X Predictors
sub_cat4 = sub_cat4[['y', 'Variant_EPOS', 'Pack_Type', 'Segment_EPOS', 'Size_Range_EPOS', 'Short_Segment', 'Promotions_1_0']]

#Creating dummy variables
sub_cat4 = pd.get_dummies(sub_cat4,drop_first=True)

#Checking whether y is balanced or not
target_count = sub_cat4.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# Zeroes are undersampled; imbalanced data

# Random Oversampling of Zeroes
count_class_0, count_class_1 = sub_cat4.y.value_counts()

# Divide by class
sub_cat4_class_0 = sub_cat4[sub_cat4['y'] == 0]
sub_cat4_class_1 = sub_cat4[sub_cat4['y'] == 1]


sub_cat4_class_0_over = sub_cat4_class_0.sample(count_class_0, replace=True,random_state=0)
sub_cat4_test_over = pd.concat([sub_cat4_class_1, sub_cat4_class_0_over], axis=0)

print('Random over-sampling:')
print(sub_cat4_test_over.y.value_counts())

sub_cat4 = sub_cat4_test_over

#Logistic regression
import statsmodels.api as sm
# random.seed(0)
X = sub_cat4.loc[:, sub_cat4.columns != 'y']
y = sub_cat4[['y']].copy()

logitm=sm.Logit(y.astype('float'),X.astype('float'))

result=logitm.fit(method='bfgs')
print(result.summary())

#Generating the confusion matrix
pred = result.pred_table(threshold=.5)
print(pred)

#Generating accuracy
acc = round((pred[0][0] + pred[1][1])/(pred[0][0] + pred[1][1] + pred[1][0] + pred[0][1])*100, 2)
print('Accuracy = ', acc, '%')
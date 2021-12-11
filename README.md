# Business-Evaluation-
Decision Making Using Market Basket Analysis

Why Association Analysis?
Association Analysis is relatively light on the math concepts and easy to explain to non-technical people. In addition, it is an unsupervised learning tool that looks for hidden patterns so there is limited need for data prep and feature engineering. It is a good start for certain cases of data exploration and can point the way for a deeper dive into the data analysis using other approaches. 
 Product recommendation algorithms account for 35% of all Amazon purchases. (McKinsey, 2013). Recommendation systems have an impact on our daily lives and basic form of recommendation system we will be looking at is Market Basket Analysis. Market Basket Analysis is a simple concept that does not necessitate advanced statistics or mathematics understanding. (Frank La, 2018). 
In Market Basket Analysis attributes like, consequent, support, confidence, lift is used for evaluation. Where support helps to study the relative frequency that the rules show up. A high support helps to show relationship. However, an instance where a low support is useful if you are trying to find “hidden” relationship. For the confidence, for product recommendation, a 50% confidence may be perfectly acceptable but in a medical situation this level may not be high enough. Lift is the ratio of the observed support to that expected if the two rules were independent. The basic rule of thumb is that a lift value close to 1 means the rules were completely independent. Lift > 1 are generally more “interesting” and could be indicative of a useful rule pattern. 
2.Application
Market Basket Analysis is used to understand the purchasing behavior of customers. It leads to effective sales and marketing. It can be used for store layout, marketing messages, maintain inventory, content placement and recommendation engines. Its application cut across affinity promotion, product placement, cross selling, fraud detection and customer behavior. Under the MBA association rule, this technique aims at searching data for frequent if-then patterns and by using a certain criterion under support and confidence to define what the most important relationships are. Support is the evidence of how frequent an item appears in the data given, as confidence is defined by how many times the if-then statements are found true. There is a third criteria called the lift and this can be used to compare the expected confidence and the actual confidence. The lift shows how many times the if-then statement is expected to be found to be true. Association rules are calculated from itemsets, which are created by two or more items. If the rules were built from the analyzing from all the possible itemsets from the dataset. 




import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
online_retail = pd.read_excel('OnlineRetail.xlsx')
online_retail.info()
online_retail.head()
online_retail = online_retail.dropna() #drop the missing values existing in the datasets
online_retail.info() #observe the information contained in the datasets
online_retail_Positive = online_retail[online_retail['Quantity']>=0] ##indeantifying the positive values from the transaction made online, since there can be orders
#cancelled online
online_retail_Positive.info()
online_retail_Positive
##for generating our MBA, we will look at the transaction within GERMANY and groupby InvoiceNo
MBA_data = (online_retail_Positive[online_retail_Positive['Country'] =="Germany"].groupby(['InvoiceNo','Description'])['Quantity']
.sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
MBA_data
def encode_units(x):
    if x <= 0:
        return 0
    if x >=1:
        return 1    

MBA_encoded = MBA_data.applymap(encode_units)
MBA_encoded
#FILTER THE ASSOCIATION BETWEEN THE ITEMS BOUGHT
MBA_filter = MBA_encoded[(MBA_encoded > 0).sum(axis=1) >=2]
MBA_filter
##applying the association rule, Apriori
from mlxtend.frequent_patterns import apriori
frequent_items = apriori(MBA_filter, min_support= 0.03,
                         use_colnames = True).sort_values('support',ascending=False).reset_index(drop=True)

frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
frequent_items
##then we find out the association between frequently bought items, using the assocuation rules
from mlxtend.frequent_patterns import association_rules
association_rules(frequent_items, metric = 'lift',
                  min_threshold = 1).sort_values('lift',ascending=False).reset_index(drop=True)

#from our above algorithm, using the association rule: we observe that there is a high association among spaceboy childrens cup & spaceboy children bowl. 
#the higher the lift value between items the higher thee association between the items, categorically if the lift value is gretaer than 1, we can conclude that there is an association between the items.











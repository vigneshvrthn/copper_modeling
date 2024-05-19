# importing the required packages
import pandas as pd
import sqlite3
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# Connect to SQLite database 
mydb = sqlite3.connect('coppermodel.db')
mycursor = mydb.cursor()

df=pd.read_sql_query('select * from copper',mydb)

#streamling app pagelayout and background and title
st.set_page_config(layout="wide")


st.title("Industrial Copper Modeling")

page_bg_img = '''
    <style>
    [data-testid=stAppViewContainer] {
        background-image: url("https://www.myfreetextures.com/wp-content/uploads/2011/06/redcopper3.jpg");
        background-size: 100% 100%; /* Cover the entire container */
        background-repeat: no-repeat; /* Ensure background image doesn't repeat */
    }
    </style>
    '''
    
st.markdown(page_bg_img, unsafe_allow_html=True)

#by using the optionmenu creating the option to view as like the page
with st.sidebar:    
    select_fun=option_menu("Menu",["Price Prediction","Status Prediction"])


# by using the threshold to select the oyion to view
if select_fun == "Price Prediction":
    

      
    cols = st.columns([2, 2, 2, 2, 2])  # Adjust the width of each column as needed

    # First column getting the input datas
    with cols[0]:
        st.markdown("<h5><span style='color:blue'>Status</span><h5>", unsafe_allow_html=True)
        status = st.selectbox("", ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised', 'Offered', 'Offerable'])
        st.markdown("<h5><span style='color:blue'>Item Type</span><h5>", unsafe_allow_html=True)
        item_type = st.selectbox("", ["SLAWR","W","S","IPL","Others","PL","WI"])
        
        

    # Second column getting the input datas
    with cols[2]:
        st.markdown("<h5><span style='color:blue'>Country</span><h5>", unsafe_allow_html=True)
        country = st.selectbox("", df["country"].unique())
        st.markdown("<h5><span style='color:blue'>No_of_days</span><h5>", unsafe_allow_html=True)
        days = st.number_input("Enter the number of days:", value=30)


        
        
       

    # Third column getting the input datas
    with cols[4]:
        st.markdown("<h5><span style='color:blue'>Enter the Quantity tons</span><h5>", unsafe_allow_html=True)
        ton = st.number_input(" ")
        st.markdown("<h5><span style='color:blue'>Enter the Thickness</span><h5>", unsafe_allow_html=True)
        thickness = st.number_input(".")
        st.markdown("<h5><span style='color:blue'>Enter the Width</span><h5>", unsafe_allow_html=True)
        width = st.number_input("-")
    
    
    
    if st.button("submit"):   # by using the button to process 
        a={"quantity_tons":[],"country":[],"status":[],"item_type":[],"thickness":[],"width":[],"days_between":[]}   #creating the dict to append and to process for dataaframe      
        a["quantity_tons"].append(ton)
        a["country"].append(country)
        a["status"].append(status)
        a["item_type"].append(item_type)        
        a["thickness"].append(thickness)
        a["width"].append(width)
        a["days_between"].append(days)
        a=pd.DataFrame(a)
        
        # if the input data is having any outlier by using the cliping to sortout them
        lis=["quantity_tons","thickness","width","days_between"]
        for i in lis:
            iqr=df[i].quantile(0.75)-df[i].quantile(0.25)
            up=df[i].quantile(0.75)+(1.5*iqr)
            low=df[i].quantile(0.25)-(1.5*iqr)
            a[i]=a[i].clip(low,up)

        # the coloumn satatus and the item_type is in catogorical so it is encoding     
        a["status"]=a["status"].map({"Wonderful":8,"Offered":7,"Offerable":6,"To be approved":5,"Revised":4,"Not lost for AM":3,"Won":2,"Lost":1,"Draft":0})
        a["item_type"]=a["item_type"].map({"SLAWR":6,"W":5,"S":4,"IPL":3,"Others":2,"PL":1,"WI":0})
        
        # spliting the datas
        from sklearn.model_selection import train_test_split
        x=df.loc[: ,list(df.columns)[:-2]+["days_between"]]
        y=df.loc[ : ,["selling_price"]]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
        
        # fitting the model by xgboost which we have analysis
        model=XGBRegressor(learning_rate=0.7)
        model.fit(x_train,y_train)
                
        # predict the input and show the price
        pre=model.predict(a.loc[: ,list(a.columns)[:]])
        st.markdown(f"## Selling price for this copper is <span style='color:green'>{pre[0]}</span>", unsafe_allow_html=True)

if select_fun == "Status Prediction":
  
   
    
      
    cols = st.columns([2, 2, 2, 2, 2])  # Adjust the width of each column as needed

    # First column getting the input datas
    with cols[0]:
        st.markdown("<h5><span style='color:blue'>Selling Price</span><h5>", unsafe_allow_html=True)
        selling_price = st.number_input("Enter the selling amt :", value=600)
        st.markdown("<h5><span style='color:blue'>Item Type</span><h5>", unsafe_allow_html=True)
        item_type = st.selectbox("", ["SLAWR","W","S","IPL","Others","PL","WI"])
        
        

    # Second column getting the input datas
    with cols[2]:
        st.markdown("<h5><span style='color:blue'>Country</span><h5>", unsafe_allow_html=True)
        country = st.selectbox("", df["country"].unique())
        st.markdown("<h5><span style='color:blue'>No_of_days</span><h5>", unsafe_allow_html=True)
        days = st.number_input("Enter the number of days:", value=30)

    

        
        
       

    # Third column getting the input datas
    with cols[4]:
        st.markdown("<h5><span style='color:blue'>Enter the Quantity tons</span><h5>", unsafe_allow_html=True)
        ton = st.number_input(" ")
        st.markdown("<h5><span style='color:blue'>Enter the Thickness</span><h5>", unsafe_allow_html=True)
        thickness = st.number_input(".")
        st.markdown("<h5><span style='color:blue'>Enter the Width</span><h5>", unsafe_allow_html=True)
        width = st.number_input("-")


    if st.button("submit"):   # by selecting the button to process
        df=pd.read_sql_query("select * from copper",mydb) # collecting the datas by sqlite3 and saved as in dataframe
        df["status"]=df["status"].map({8:1,7:1,6:1,5:1,4:0,3:1,2:1,1:0,0:0})    # maping the datas by 0 and 1 by its nature
        
        #creating the dict and append the input datas and converting into dataframe
        a={"quantity_tons":[],"country":[],"item_type":[],"thickness":[],"width":[],"selling_price":[],"days_between":[]}        
        a["quantity_tons"].append(ton)
        a["country"].append(country)        
        a["item_type"].append(item_type)        
        a["thickness"].append(thickness)
        a["width"].append(width)
        a["selling_price"].append(selling_price)
        a["days_between"].append(days)
        a=pd.DataFrame(a)


        ## if the input data is having any outlier by using the cliping to sortout them 
        lis=["quantity_tons","thickness","width","days_between","selling_price"]
        for i in lis:
            iqr=df[i].quantile(0.75)-df[i].quantile(0.25)
            up=df[i].quantile(0.75)+(1.5*iqr)
            low=df[i].quantile(0.25)-(1.5*iqr)
            a[i]=a[i].clip(low,up)
        
        # the coloumn  item_type is in catogorical so it is encoding     
        a["item_type"]=a["item_type"].map({"SLAWR":6,"W":5,"S":4,"IPL":3,"Others":2,"PL":1,"WI":0})
        
        # spliting the datas
        x = df.loc[:, list(df.columns)[:2] + list(df.columns)[3:]]
        y = df.loc[:, ["status"]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
        
        #fitting the model by random classifier
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        
        #predicting the input and shown the output
        pre=model.predict(a.loc[: ,list(a.columns)[:]])
        if pre==1:
            st.markdown(f"## Status for this copper is <span style='color:green'>WON</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"## Status for this copper is <span style='color:red'>LOST</span>", unsafe_allow_html=True)
        


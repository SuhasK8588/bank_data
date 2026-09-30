import pandas as pd
import sqlite3 
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import streamlit as st
import datetime
import os
import kagglehub

class Bank:
    def __init__(self):
        self.db_name = f"bank.db"
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.create_table()
        self.scaler = None
        self.status = None
        self.amount = None
        self.term = None
        self.feature_columns = []
    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS Bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT,
            Age TEXT,
            Income TEXT,
            Networth TEXT,
            Adarcard_number REAL
        )'''
        self.conn.execute(query)
        self.conn.commit()

    def load_data(self):
        query = "SELECT * FROM Bank"
        return pd.read_sql_query(query, self.conn)
    def train_models(self):
        path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
        print("Path to dataset files:", path)
        file_path = os.path.join(path, 'loan_approval_dataset.csv')
        df = pd.read_csv(file_path)
        if 'loan_id' in df.columns:
            df = df.drop(['loan_id'], axis=1)
        df.columns = df.columns.str.strip()
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
     
        
        x = df.drop(["loan_status", "loan_amount", "loan_term"], axis=1)
               
        
        
        y_status = df["loan_status"].map({'Approved': 1, 'Rejected': 0})
        y_amount = df["loan_amount"]
        y_term = df["loan_term"]
        x = pd.get_dummies(x, drop_first=True,dtype=int)
        self.feature_columns = x.columns.tolist()
        self.scaler = StandardScaler()
        x_scale = self.scaler.fit_transform(x)
        self.status=RandomForestClassifier()
        self.status.fit(x_scale,y_status)
        self.amount=RandomForestRegressor()
        self.amount.fit(x_scale,y_amount)
        self.term=RandomForestRegressor()
        self.term.fit(x_scale,y_term)
        

    def predict(self,name,no_of_dependents,income_annum,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value
                ,bank_asset_value,education,self_employed):
        
        if self.scaler is None:
            return {"Error": "Models haven't been trained yet! Run train_models() first."}
        
        edu_not_graduate = 1 if int(education) == 0 else 0
        emp_yes = int(self_employed)
        input_data = {
            "no_of_dependents": int(no_of_dependents),
            "income_annum": float(income_annum),
            "cibil_score": float(cibil_score),
            "residential_assets_value": float(residential_assets_value),
            "commercial_assets_value": float(commercial_assets_value),
            "luxury_assets_value": float(luxury_assets_value),
            "bank_asset_value": float(bank_asset_value),
            "education_Not Graduate": edu_not_graduate,
            "self_employed_Yes": emp_yes
        }

        user_df = pd.DataFrame([input_data])[self.feature_columns]
        user_scaled = self.scaler.transform(user_df)
        
        status_pred = self.status.predict(user_scaled)[0]
        
        if status_pred == 1:
            amount_pred = self.amount.predict(user_scaled)[0]
            interest_pred = self.term.predict(user_scaled)[0]
            return {
                "Name": name,
                "Approved": "Yes",
                "Loan Amount Approved": f"₹{amount_pred:,.2f}",
                "Term": f"{interest_pred:.2f}%"
            }
        else:
            return {
                "Name": name,
                "Approved": "No",
                "Loan Amount Approved": "₹0.00",
                "Term": "N/A"
            }
class BankTransaction:
    def __init__(self,username):
        self.db_name=f"{username}.Bank_transation.db"
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.create_table()
    def get_current_balance(self):
        query = "SELECT Balance FROM Bank_transation ORDER BY id DESC LIMIT 1"
        cursor = self.conn.execute(query)
        row = cursor.fetchone()
        return row[0] if row else 0.0
  
    def create_table(self):
        query = '''
        CREATE TABLE IF NOT EXISTS Bank_transation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            Amount REAL,
            Type TEXT,
            Balance REAL
        )'''
        self.conn.execute(query)
        self.conn.commit()

    def load_data(self):
        query = "SELECT * FROM Bank_transation"
        return pd.read_sql_query(query, self.conn)
    def Deposit(self,date,amount):
        if amount:
            current_balance = self.get_current_balance() + amount
            query = "INSERT INTO Bank_transation(Date, Amount,Type,Balance) VALUES(?, ?, ?, ?)"
            self.conn.execute(query, (str(date),amount, "Income",current_balance))
            self.conn.commit()
            st.success("Your income activity has been saved!")
        else:
            st.error("Please enter a valid amount.")

    def withdraw(self, amount,date):
        if self.get_current_balance()>amount:
            current_balance = self.get_current_balance() - amount
            query = "INSERT INTO Bank_transation(Date,Amount,Type,Balance) VALUES(?, ?, ?, ?)"
            self.conn.execute(query, (str(date),amount , "withdraw",current_balance))
            self.conn.commit()
            st.success("Your expense activity has been saved!")
        else:
            st.error("Please enter a valid amount.")
    def statement(self):
        return self.load_data()
    def prediction(self,date):
        df=self.load_data()
        if len(df)<5:
            return "not enough to predict the data"
        
        df_ml = df.copy()
        df_ml['Date_Ordinal'] = pd.to_datetime(df_ml['Date']).apply(lambda x: x.toordinal())
        
        X = df_ml[['Date_Ordinal']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)        
        Y_type = df_ml["Type"]
        Y_amount = df_ml["Amount"]
        
        model_type = RandomForestClassifier(n_estimators=100, random_state=42)
        model_type.fit(X_scaled, Y_type)
        
        
        model_amount = RandomForestRegressor(n_estimators=100, random_state=42)
        model_amount.fit(X_scaled, Y_amount)
        
        target_ordinal = pd.to_datetime(date).toordinal()
        target_scaled = scaler.transform(np.array([[target_ordinal]]))
        
        pred_type = model_type.predict(target_scaled)[0]
        pred_amount = model_amount.predict(target_scaled)[0]
        
        return f"prediction on {date}:{pred_type}:{pred_amount} rs"
        

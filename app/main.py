#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:11:57 2023

@author: frodo
"""

import pandas as pd
import lightgbm as lgb
import pickle
import gdown  # Import gdown for file downloading
from datetime import datetime, timedelta
from fastapi import FastAPI
from starlette.responses import JSONResponse


app = FastAPI()

#load predictive model
predictive_model = lgb.Booster(model_file='../models/lightgbm.txt')

#load forecasting model
with open('../models/holtwinters.pkl', 'rb') as f:
    forecasting_model = pickle.load(f)

@app.get("/")
def root():
    return {
        "description": "This is a sales forecasting API with several endpoints to predict sales volume.",
        "endpoints": {
            "/": "Brief description of the project and endpoints.",
            "/health/": "Check the health status of the API.",
            "/sales/national/": "Get the next 7 days sales volume forecast for an input date.",
            "/sales/stores/items/": "Get predicted sales volume for an input item, store, and date."
        },
        "Github repo": "https://github.com/frodorocky/adv_mla_ass_2_hw_api"
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'predictive and forecasting model are all ready to go!'

@app.get("/sales/national/")
def sales_national(date: str):
    
    # Convert input_date string to a date object
    input_date = datetime.strptime(date, "%Y-%m-%d").date()
        
    # Define the start date
    start_date = datetime.strptime("2015-04-18", "%Y-%m-%d")
    
    # Calculate the difference in days
    days_difference = (input_date - start_date).days + 7
    
    # Forecast using the days_difference as steps
    results = forecasting_model.forecast(steps=days_difference)

    # Extract the prediction for the next 7 days
    forecast_next_7_days = results[-7:]
    
    # Mocking data for next 7 days forecast
    forecast = {str(input_date + timedelta(days=i)): forecast_next_7_days[i] for i in range(7)}
    
    return JSONResponse(forecast)

@app.get("/sales/stores/items/")
def sales_stores_items(date: str, item_id: str, store_id: str):
    
    # Convert input_date string to a date object
    date = datetime.strptime(date, "%Y-%m-%d").date()
    
    # load data
    url = "https://drive.google.com/uc?export=download&id=1MxzLINcFcUrdoTQZGLsBcrPDpBM0rbEL"
    output = 'predictive_test_data.csv'
    gdown.download(url, output, quiet=False)

    predict_df = pd.read_csv(output)
    
    filtered_rows = predict_df[
        (predict_df['date'] == date) &
        (predict_df['item_id'] == item_id) &
        (predict_df['store_id'] == store_id)
    ]
    
    # Mocking data for sales volume prediction
    pred = predictive_model.predict(filtered_rows)
    
    return JSONResponse(pred)



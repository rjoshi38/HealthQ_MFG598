import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

df = pd.read_csv('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/processed_data/processedfile.csv')
df2 = pd.read_csv('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/dummy_patient_appointment_data.csv') 
df2["date"] = pd.to_datetime(df2["date"])
df3 = pd.read_csv('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/processed_data/testData.csv') 
plotDir = '/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Plots'

X = df.drop('waitTime', axis=1)
y = df['waitTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#LR
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
y_pred = linearModel.predict(X_test)

lrmae = mean_absolute_error(y_test, y_pred)
lrmse = mean_squared_error(y_test, y_pred)
lrrmse = np.sqrt(lrmse)
lrr2 = r2_score(y_test, y_pred)

#XGB
xgbModel = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgbModel.fit(X_train, y_train)
y_pred = xgbModel.predict(X_test)

xgmae = mean_absolute_error(y_test, y_pred)
xgmse = mean_squared_error(y_test, y_pred)
xgrmse = np.sqrt(xgmse)
xgr2 = r2_score(y_test, y_pred)

#RF
rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
rfModel.fit(X_train, y_train)
y_pred = rfModel.predict(X_test)

rfmae = mean_absolute_error(y_test, y_pred)
rfmse = mean_squared_error(y_test, y_pred)
rfrmse = np.sqrt(rfmse)
rfr2 = r2_score(y_test, y_pred)


#LGBM
lgbModel = lgb.LGBMRegressor(objective='regression', metric='l2', random_state=42)
lgbModel.fit(X_train, y_train)
y_pred = lgbModel.predict(X_test)

lgbmae = mean_absolute_error(y_test, y_pred)
lgbmse = mean_squared_error(y_test, y_pred)
lgbrmse = np.sqrt(lgbmse)
lgbr2 = r2_score(y_test, y_pred)

#Compare
modelResults = {"Model": [], "MAE": [], "MSE": [], "RMSE": [], "R-squared": []}
def addResults(modelName, mae, mse, rmse, r2):
    modelResults["Model"].append(modelName)
    modelResults["MAE"].append(mae)
    modelResults["MSE"].append(mse)
    modelResults["RMSE"].append(rmse)
    modelResults["R-squared"].append(r2)
addResults("Linear Regression", lrmae, lrmse, lrrmse, lrr2)
addResults("XGBoost", xgmae, xgmse, xgrmse, xgr2)
addResults("Random Forest", rfmae, rfmse, rfrmse, rfr2)
addResults("Light GBM", lgbmae, lgbmse, lgbrmse, lgbr2)
resultsdf = pd.DataFrame(modelResults)
print("Model performance comparison: ")
print(resultsdf)

#Recommendation
modelScores = {}
for index, row in resultsdf.iterrows():
    modelName = row['Model']
    mae = row['MAE']
    mse = row['MSE']
    rmse = row['RMSE']
    r2 = row['R-squared']
#Score to predict best model with low error, high r-squared values
    modelScores[modelName] = {"MAE":mae, "MSE":mse, "RMSE":rmse, "R-squared":r2, "score":((mae+mse)-r2)}
bestModel = min(modelScores, key=lambda x:modelScores[x]["score"])
print("Modl Scores: ")
for model, scores in modelScores.items():
    print(f"{model} -> MAE: {scores['MAE']:2f}, MSE: {scores['MSE']:2f}, RMSE: {scores['RMSE']:2f}, R-squared: {scores['R-squared']:2f}, Score: {scores['score']:2f}")
print(f"\nRecommended model is {bestModel}, based on lowest error and highest R-squared value.")


#Dashboard

#Load dataset with caching
@st.cache_data
def dummyDataload():
    df2 = pd.read_csv('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/dummy_patient_appointment_data.csv')
    df2['date'] = pd.to_datetime(df2['date'])
    return df2

@st.cache_data
def testDataload():
    df3 = pd.read_csv('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/processed_data/testData.csv')
    return df3

def modelandtools():
    encoder = joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/Files/encoder.pkl')
    scaler = joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/Files/scaler.pkl')
    finalFeatures = joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/Files/finalFeatures.pkl')
    models={"Linear Regresion": joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Models/linear_regression.pkl'),
            "XGBoost": joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Models/xgboost.pkl'),
            "Random Forest": joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Models/random_forest.pkl'),
            "Light GBM": joblib.load('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Models/random_forest.pkl')}
    return encoder, scaler, finalFeatures, models

plotDir = ('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ML Plots')
data = dummyDataload()
testData = testDataload()
encoder, scaler, finalFeatures, models = modelandtools()

#Title
st.title("HealthQ Dashboard")

#Creating tabs
tabs = st.tabs(['About', 'Historical Trends', 'ML Model Comparison', 'Wait Time Prediction'])

#Tab0 - About 
with tabs[0]:
    st.header('About')
    st.image('/Users/rohitjoshi/University/Fall 2024/MFG 598/Project/ASU_Logo', width=200)
    st.markdown("""
    ##Welcome viewer!

    While working with ASU Campus Health Services, it was highlighted that patient wait times are a vital parameter that needs to be studied to provide better and faster service to the students. This metric also helps the leadership in planning and allocating resources as needed.
    Hence, I took on the task to devise a dashboard which will not only show patient appointment wait time trends, but also illustrate predicted wait time, based on user inputs.
    
    Project Overview:
    I took the following path for this project:
    1. **Data Creation**: Since actual data cannot be used due to HIPAA violation risk, I created a dummy data, covering all relevant fields which impact wait time.
    2. **Pre-Processing**: Once this was created, performed normalization, scaling, and encoding to ensure the dataset is in a standardized, and usable format.
    3. **Data Splitting**: Processed dataset was split as per 80-20 division for Training and Testing.
    4. **Model Training**: Trained 4 ML models - Linear Regression, XGBoost, Random Forest Regressor, LightGBM.
    5. **Performance Comparison**: Compared the performance of these models in a tabular form, along with graphs.
    6. **Recommendation**: Based on these metrics, recommendation was made for the optimal ML model to use for prediction in this case.
    7. **Dashboard Development**: Built this dashboard to showcase the following:
    **Historical Trends** Helps to assess hisotrical trend of patient wait times for one of all campuses and a particular date range. 
    **ML Model Comparison**: Displays performance metrics of all ML models trained with recommendation. Graph for actual vs predicted values for the recommended model is also shown.
    **Prediction**: Dynamic, interactive dashboard which accepts user inputs and gives out predicted wait time for selected parameters.

    This application is designed to provide insights and empower leadership in improving clinic performance.

    Resource:
    You can explore the code and files for this project on [My Github Profile](https://github.com/rjoshi38).
    """)

#Tab1 - Historical Trends Section
with tabs[1]:
    st.header("Historical Wait Time Trends")
    campusOptions = ['All'] + list(df2['campus'].unique())
    campus = st.selectbox('Campus', campusOptions)
    
    startDate, endDate = st.date_input('Date Range', value=(df2['date'].min(), df2['date'].max()))
    filteredData = df2[(df2['date'] >= pd.to_datetime(startDate)) & (df2['date'] <= pd.to_datetime(endDate))]

    if campus != 'All':
        filteredData = filteredData[filteredData['campus'] == campus]
        
    filteredData['Wait Time'] = (filteredData['waitTime']/60)
    
    fig = px.bar(filteredData, x='date', y='waitTime', color='campus')
    st.plotly_chart(fig)

#Tab2 - ML Model Comparison Section
with tabs[2]:
    st.header("ML Model Comparison")
    resutlsdf = pd.DataFrame(modelResults) #recall ML performance metrics
    st.subheader("Model Performance Metrics")
    st.table(resultsdf)

    st.success(f"Recommended Model: {bestModel}") #recall recommendation

    bestModelMetrics = resultsdf[resultsdf['Model'] == bestModel].iloc[0]
    st.markdown(f"""
        ***Recommended Model Perf Metrics**:
        - **MAE**: {bestModelMetrics['MAE']:.2f}
        - **MSE**: {bestModelMetrics['MSE']:.2f}
        - **RMSE**: {bestModelMetrics['RMSE']:.2f}
        - **R2**: {bestModelMetrics['R-squared']:.2f}
        """)

    st.subheader(f"Actual Vs Predicted Values - {bestModel}") #Robust model with no hard coded recommendation
    plotFile = os.path.join(plotDir, f"{bestModel.lower().replace(' ', '_')}_plot.png")
    if os.path.exists(plotFile):
        st.image(plotFile, caption=f"Actual Vs Predicted - {bestModel}")
    else:
        st.error(f"Plot for {bestModel} missing")

#Tab3 - Predicted Wait Time Section
with tabs[3]:
    st.header("Wait Time Prediction")
    campusColumns = [col for col in df3.columns if col.startswith('campus_')]
    campusInput = st.selectbox('Campus', options = [col.replace('campus_','') for col in campusColumns])

    selDate = st.date_input('Date', value = pd.to_datetime('2023-01-1'))
    defaultTime = pd.Timestamp('08:00:00').time()
    selTime = st.time_input('Time', value = defaultTime)

    if selTime <pd.Timestamp('08:00:00').time() or selTime > pd.Timestamp('17:00:00').time():
        st.error('Time must be between 8am and 5pm')
    else:
        apptColumns = [col for col in df3.columns if col.startswith('appointmentType_')]
        apptTypeInput = st.selectbox('Appt Type', options=[col.replace('appointmentType_', '') for col in apptColumns])

        urgencyColumns = [col for col in df3.columns if col.startswith('urgency_')]
        urgencyInput = st.selectbox('Urgency', options=[col.replace('urgency_', '') for col in urgencyColumns])

        providerCount = st.slider('Provider Count', 1, 15, 5)
        clinicCount = st.slider('Clinic Load', 10, 200, 100)

        testData = testData.drop(columns=['waitTime'], errors='ignore')

        #using df3, testing dataset for prediction
        input_array = np.zeros(len(finalFeatures))
        for feature in finalFeatures:
            if f'campus_{campusInput}' in finalFeatures:
                input_array[finalFeatures.index(f'campus_{campusInput}')] = 1
            if f'appointmentType_{apptTypeInput}' in finalFeatures:
                input_array[finalFeatures.index(f'appointmentType_{apptTypeInput}')] = 1
            if f'urgency_{urgencyInput}' in finalFeatures:
                input_array[finalFeatures.index(f'urgency_{urgencyInput}')] = 1
        if 'providerLoad' in finalFeatures:
            input_array[finalFeatures.index(f'providerLoad')] = providerCount
        if 'clinicLoad' in finalFeatures:
            input_array[finalFeatures.index(f'clinicLoad')] = clinicCount

        recommendedModel = models[f'{bestModel}']
        predictedWaitTime = recommendedModel.predict([input_array])[0]
        st.success(f'Predicted Wait Time with selected conditions: {predictedWaitTime:.2f} mins')

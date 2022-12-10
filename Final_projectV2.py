# import libraries
import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import hiplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error
#sns.set_style("darkgrid")

# get data
df = pd.read_csv('exit_loop.csv')
# st.write(df)

#labels_df = df.select_dtypes(include='float64').columns  # feel free to change this
# st.write(labels_df)

st.title("Exploring the Relationship between Crash Frequency and Deceleration Lanes")

st.markdown("""
    The data contains information on the number of crashes on deceleration lanes on freeways.

    ## Problem Statement

    Speed Change Lanes (SCL) are usually crash-prone because of the conflicts that occurs when entering or exiting a freeway 
    (usually 70-85mph roadway) into a loop or a ramp. Most crashes on the freeways usually occur at the area of influence of 
    the SCL due to the deceleration or acceleration of vehicles merging into the freeway or leaving the freeway. 
    Although, the problems within SCL are known to all, there has been no conclusive research on the lengths of the SCL 
    (Some research concludes that decrease in SCL would reduce crashes, while others concludes that increase in SCL would increase crashes).

    This project would aim to provide more insights to the relationship between SCL and road crashes. 
    This would be done by evaluating the relationship between the length of the SCL and crash frequency or crash rate. Other variables 

    Conversely, this project would help policymakers in improving and upgrading standards and specification relating to SCL and freeways.

    Note: The models does not seem to perform well because of the nature of crash data with many zeros.

    ### Data Description
    __Ln MainlineAADT and Ln Ramp AADT:__ The log transform of the traffic volume on the mainline and the ramp

    __Deceleration lane length:__ The length of the exit Speed Change Lane (SCL) measured

    __Crash Frequency:__ The total number of crashes on the speed change lane in a year
   


    """)

#######################
## streamlit sidebar ##
#######################
tab1,tab2 = st.tabs(["Data visualization","Machine Learning"])
with tab1:
    st.sidebar.title("""
    # Visualizing the Data
    Selecting the different parameters that contribute to an increase in crashes.
    """)

    # allow user to choose which portion of the data to explore
    y_axis_choice = st.sidebar.selectbox("y axis", ["Crash Frequency"])
    hue_dia = st.sidebar.selectbox('hue: ', df.select_dtypes(include = "object").columns)

    st.header("Joint, Categorical, Violin Plot, Box Plot, Bar Plot and HiPlot")
    sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        (
            "Joint Plot", #First option in menu
            "Categorical Plot",
            "Violin Plot",
            "Box Plot",
            "Bar Plot",
            "HiPlot"
        )
    )

    fig, ax = plt.subplots()


    if sd == "Joint Plot":
        f = True
        x_axis_choice = st.sidebar.selectbox('x axis: ', df.select_dtypes(include = "number").columns)
        sns.jointplot(data = df, x = x_axis_choice, y = y_axis_choice, hue = hue_dia, ax = ax)
        st.pyplot(plt.gcf())

    elif sd == "Categorical Plot":
        f = True
        x_axis_choice = st.sidebar.selectbox('x axis: ', df.select_dtypes(include = "number").columns)
        sns.catplot(data = df, x = x_axis_choice  , y = y_axis_choice, hue = hue_dia, split = True, palette = "pastel", ax = ax)
        st.pyplot(plt.gcf())
        

    elif sd == "Violin Plot":
        f = False
        x_axis_choice = st.sidebar.selectbox('x axis: ', df.select_dtypes(include = "object").columns)
        sns.violinplot(data = df, x = x_axis_choice, y = y_axis_choice, ax = ax)
        st.pyplot(plt.gcf())


    elif sd == "Box Plot":
        f = False
        x_axis_choice = st.sidebar.selectbox('x axis: ', df.select_dtypes(include = "object").columns)
        sns.boxplot(data = df, x = x_axis_choice, y = y_axis_choice, ax = ax)
        st.pyplot(plt.gcf())

    elif sd == "Bar Plot":
        f = False
        x_axis_choice = st.sidebar.selectbox('x axis: ', df.select_dtypes(include = "object").columns)
        sns.barplot(data = df, x = x_axis_choice, y = y_axis_choice, ax = ax)
        st.pyplot(plt.gcf())



    elif sd == 'HiPlot':

        variables = st.multiselect("select variables",["Crash Frequency", "Ln.MainlineAADT", "Ln.RampAADT", 
            "Decel_length","Lane_width_ramp", "Laneconfig_code", "CrossroadConfig_Code"], 
            ["Ln.MainlineAADT", "Ln.RampAADT", "Decel_length", "Crash Frequency"])

    # If the mistakenly delete all columns
        if(len(variables) == 0):
            st.write("""
                ### Please select one or more columns!
                """)

        else:
            # New Dataset created based on the column selection
            Hip = df[variables]

            # Plotting the data using Hiplot
            xp = hiplot.Experiment.from_dataframe(Hip)
            xp.to_streamlit(key="hip").display()


    X = df[["Ln.MainlineAADT", "Ln.RampAADT", "Decel_length", "Lane_width_ramp", "Laneconfig_code", "CrossroadConfig_Code"]]
    y = df[["Crash Frequency"]]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


    def correlation(df,col1,col2):
        rmse=np.sqrt(mean_squared_error(df[col1], df[col2]))
        r=stats.pearsonr(df[col1], df[col2])
        r_square= pow(r[0], 2)
        return rmse,r,r_square
 
    def create_data(columns, label):

        data = df[columns]
        label = df[label]

        # convert the data into series
        data = data.to_numpy()
        label = label.to_numpy()

        return data, label

    def create_newFeat(data, feat_num):

        N = data.shape[0]    
        # arrays and vectors to store the new data
        x_data = np.zeros((N-feat_num, feat_num))
        yVec = np.zeros(N-feat_num)

        # loop through the data to extract the features
        for i in range(N-feat_num):
            for j in range(feat_num):
                x_data[i, j] = data[i+j]
                yVec[i] = data[i+feat_num]
        return x_data, yVec

    def use_ML(data_train, label_train,data_test, est):
        """"Run Machine Learning Model.""" 

        if est == 'DT':

            rng = np.random.RandomState(42)
            #regr = AdaBoostRegressor(
                #DecisionTreeRegressor(min_samples_leaf=leaf, max_features=max_featD), n_estimators=300, random_state=rng)
            regr = DecisionTreeRegressor(min_samples_leaf=leaf, max_features=max_featD)
            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

        elif est == 'RF':
            regr = RandomForestRegressor(n_estimators=n_est, max_features=max_featR)

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

        elif est == "BR":
            regr = BayesianRidge()

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)
            
        elif est == 'LR':
            regr = LinearRegression()

            # fit the data and label
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

            y_pred = np.ravel(y_pred)

        elif est == "SVC":
            rng = np.random.RandomState(42)
            regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)


        elif est == 'NuSVR':
            rng = np.random.seed(42)
            regr = make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1))
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

        elif est == "NB":
            regr = GaussianNB()
            regr.fit(data_train, label_train)

            # Make prediction of the model
            y_pred = regr.predict(data_test)

        elif est == "GR":
            rng = np.random.seed(42)
            regr = GradientBoostingRegressor()
            regr.fit(data_train, label_train)

             # Make prediction of the model
            y_pred = regr.predict(data_test)


        return y_pred
   
#  ---------------------------------------------------------------------------------------------- #
#  Machine Learning
#  ---------------------------------------------------------------------------------------------- #

with tab2:
    st.write("""## MACHINE LEARNING APPLICATION""")
    st.write("""##### Select Machine Learning Regression""")

    ml = st.selectbox('Estimators: ', ['Decision Tree', 'Random Forest', "BayesianRidge", 
        'Linear Regression', "SVC", "NuSVR", "Naive Bayes", "Gradient Boosting"])
    col1, col2 = st.columns(2)
    with col1:
        features = st.multiselect('Select features:',
            ["Ln.MainlineAADT", "Ln.RampAADT", "Decel_length", "Lane_width_ramp", "Laneconfig_code", "CrossroadConfig_Code"], 
            ["Ln.MainlineAADT", "Ln.RampAADT", "Decel_length"])

    with col2:
        label_select = st.selectbox('Select label: ', ["Crash Frequency"])

    # If a mistake is made such that no feature is selected
    max_ft = len(features)
    if max_ft == 0:
        st.write('''### Ooops! Please select atleast one feature ''')
    else:

        # Conditioning the dataset for the Machine Learning
        #data, label = create_data(df, features, label_select)

        # Running the Individual Estimators
        if ml == 'Random Forest':

            ml_col1, ml_col2 = st.columns(2)
            with ml_col1:
                # Making a default value for the slider after rendering
                if max_ft == 1:
                    max_featR = max_ft  # This will be used as the hyperparameter for Max Features
                else: 
                    val = max_ft-1
                    max_featR = st.slider('Max Feature:', 1, max_ft, val)

            with ml_col2:
                n_est = st.slider('Number of Trees:', 20, 100, 50)

            # Running the Random Forest Regressor Estimator
            y_pred = use_ML(X_train[features], y_train,X_test[features], est='RF')

        elif ml == 'Decision Tree':
            ml_col_DT1, ml_col_DT2= st.columns(2)
            with ml_col_DT1:
                # Making a default value for the slider after rendering
                if max_ft == 1:
                    max_featD = max_ft  # This will be used as the hyperparameter for Max Features
                else: 
                    val = max_ft-1
                    max_featD = st.slider('Max Feature:', 1, max_ft, val)

            with ml_col_DT2:
                leaf = st.slider('Number of Leafs:', 1, 10, 1)

            # Running the Decision Tree Estimator
            y_pred = use_ML(X_train[features], y_train,X_test[features], est='DT')

        elif ml == "BayesianRidge":
           y_pred = use_ML(X_train[features], y_train, X_test[features], est = "BR")

        elif ml == 'Linear Regression':
            # Running the Linear Regressor Estimator
            y_pred= use_ML(X_train[features], y_train,X_test[features], est='LR')

        elif ml == "SVC":
            y_pred= use_ML(X_train[features], y_train,X_test[features], est='SVC')


        elif ml == "NuSVR":
            y_pred = use_ML(X_train[features], y_train,X_test[features], est='NuSVR')

        elif ml == "Naive Bayes":
            y_pred = use_ML(X_train[features], y_train,X_test[features], est='NB')

        elif ml == "Gradient Boosting":
            y_pred = use_ML(X_train[features], y_train,X_test[features], est='GR')
       
        
        # Select the Range of Values to Plot
        st.subheader("Plotting The Actual and predicted Crashes")

        start, end = st.slider(
            'Select The Range of Values to plot', 
            0, df.shape[0], [0, 630])

        df_mpg=pd.DataFrame()
        df_mpg=X_test[features]
        df_mpg['Crash Predicted']=y_pred
        df_mpg['Crash Observed']=y_test
        
        #MM.visulaization_error()
        rmse_cal,r_cal,r_square_cal=correlation(df_mpg,'Crash Observed','Crash Predicted')
        st.write("""###### Model Performance:""")
        df_score = pd.DataFrame(np.array([rmse_cal**2, rmse_cal,r_square_cal]).reshape(1, -1), columns=('MSE', 'RMSE','R2'))
        st.table(df_score)
        scatter = alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Crash Observed', y='Crash Predicted').interactive()
        reg_line=alt.Chart(df_mpg).properties(width=350).mark_circle(size=100).encode(x='Crash Observed', 
            y='Crash Predicted').transform_regression('Crash Observed','Crash Predicted').mark_line()
        scatter+reg_line
        
    #############################################################################################
    #END
    #############################################################################################


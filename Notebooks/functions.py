from huggingface_hub import InferenceClient

def chat_completion(input):
    from dotenv import load_dotenv
    import os

    # Load environment variables from a .env file
    load_dotenv(r'C:\Users\dalre\Documents\GitHub\aviation_investigation_model\.env')

    # Retrieve the API key from the environment variables
    api_key = os.getenv("HUGGINGFACE_API_KEY")    
    client = InferenceClient(
        "microsoft/Phi-3-mini-4k-instruct",
    token=api_key,
    )

    message = client.chat_completion(
        messages=[{"role": "user", "content": input}],
        max_tokens=500,
        stream=False,
    )
    
    return message.choices[0].message.content

    

#If using meta llama 3.1 api

# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     "mistralai/Mistral-7B-Instruct-v0.1",
#     token="hf_XXXX",
# )

# for message in client.chat_completion(
# 	messages=[{"role": "user", "content": "What is the capital of France?"}],
# 	max_tokens=500,
#   temperature=o.5 (can change this number from 0 to 1.9, higher the number more random the ans)
# 	stream=True,
# ):
#     print(message.choices[0].delta.content, end="")

import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def load_and_preprocess_data(file_path):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path)
    
    # Extracting letter and year from Report_ID and Risk_Classification
    df['Department'] = df['Report_ID'].str[0]
    df['Year'] = df['Report_ID'].str[1:3]
    df['Report_ID'] = df['Report_ID'].str[3:6]
    
    # Extracting letter from Risk_Classification
    df['Risk_Classification_Number'] = df['Risk_Classification'].str[1]
    df['Risk_Classification'] = df['Risk_Classification'].str.extract('(\w)')
    
    # Reordering the Dataframe
    df = df[['Report_ID', 'Department', 'Year', 'Risk_Classification', 'Risk_Classification_Number', 'Sector', 'Month', 'Time', 'Safety_Hazard']]
    
    return df


def visualize_risk_classification_by_department(df, department, title):
    filtered_df = df[df['Department'] == department]

    grouped_df = filtered_df[filtered_df['Risk_Classification'] == 'E'].groupby(['Year', 'Risk_Classification_Number']).size().reset_index(name='Count')

    fig = go.Figure()

    for risk_classification_number in grouped_df['Risk_Classification_Number'].unique():
        temp_df = grouped_df[grouped_df['Risk_Classification_Number'] == risk_classification_number]
        fig.add_trace(go.Bar(
            x=temp_df['Year'],
            y=temp_df['Count'],
            name='Risk Classification Number: ' + str(risk_classification_number)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Count',
        hovermode='x',
        barmode='stack',
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'All',
                        'method': 'restyle',
                        'args': [{'visible': [True] * 3}, {'title': 'All Risk Classifications'}]
                    },
                    {
                        'label': 'E',
                        'method': 'restyle',
                        'args': [{'visible': [True, False, False]}, {'title': f'Occurrences of E in Department {department} by Year'}]
                    },
                    {
                        'label': 'S',
                        'method': 'restyle',
                        'args': [{'visible': [False, True, False]}, {'title': f'Occurrences of S in Department {department} by Year'}]
                    },
                    {
                        'label': 'M',
                        'method': 'restyle',
                        'args': [{'visible': [False, False, True]}, {'title': f'Occurrences of M in Department {department} by Year'}]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'active': 0
            }
        ]
    )

    fig.show()


def visualize_safety_hazards_across_departments(df):
    fig = go.Figure()
    departments = df['Department'].unique()

    for dept in departments:
        filtered_df = df[df['Department'] == dept]
        grouped_df = filtered_df.groupby(['Year', 'Safety_Hazard']).size().reset_index(name='Count')

        for hazard in grouped_df['Safety_Hazard'].unique():
            temp_df = grouped_df[grouped_df['Safety_Hazard'] == hazard]
            fig.add_trace(go.Bar(
                x=temp_df['Year'],
                y=temp_df['Count'],
                name=f'{hazard} in {dept}',
                visible=True  # Initially make all traces visible
            ))

    # Create the dropdown buttons
    buttons = []

    # Add the "All" button first
    buttons.append(
        {
            'label': 'All',
            'method': 'restyle',
            'args': [{'visible': [True] * len(fig.data)}, {'title': 'Safety Hazards in All Departments by Year'}]
        }
    )

    # Add buttons for each department
    for dept in departments:
        visibility = []
        for trace in fig.data:
            if f'in {dept}' in trace.name:
                visibility.append(True)
            else:
                visibility.append(False)

        buttons.append(
            {
                'label': dept,
                'method': 'restyle',
                'args': [{'visible': visibility}, {'title': f'Safety Hazards in Department: {dept} by Year'}]
            }
        )

    fig.update_layout(
        title='Safety Hazards in All Departments by Year',
        xaxis_title='Year',
        yaxis_title='Count',
        barmode='stack',
        hovermode='x',
        updatemenus=[
            {
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
            }
        ]
    )

    fig.show()



def chi_square_test(df, col1, col2):
    chi2_table = pd.crosstab(df[col1], df[col2])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(chi2_table)
    return chi2_stat, p_val, dof, expected



def anova_test(df, dependent_var, independent_var):
    # Ensure the dependent variable is numeric
    df[dependent_var] = pd.to_numeric(df[dependent_var], errors='coerce')
    df = df.dropna(subset=[dependent_var, independent_var])
    
    # Perform ANOVA
    model = ols(f'{dependent_var} ~ C({independent_var})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table



def t_test_between_departments(df, dependent_var, dept1, dept2):
    # Ensure the dependent variable is numeric
    df[dependent_var] = pd.to_numeric(df[dependent_var], errors='coerce')
    df = df.dropna(subset=[dependent_var])
    
    # Extract data for the departments
    dept1_data = df[df['Department'] == dept1][dependent_var]
    dept2_data = df[df['Department'] == dept2][dependent_var]
    
    # Perform the t-test
    t_stat, p_val = stats.ttest_ind(dept1_data, dept2_data, equal_var=False)
    return t_stat, p_val



def convert_to_minutes(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except ValueError:
        return None  # Handle cases where time is not properly formatted

def correlation_test(df, time_col, dependent_var):
    # Ensure the dependent variable is numeric
    df[dependent_var] = pd.to_numeric(df[dependent_var], errors='coerce')
    
    # Convert time to minutes since midnight
    df['Time_in_Minutes'] = df[time_col].apply(convert_to_minutes)
    
    # Drop NaN values
    df = df.dropna(subset=['Time_in_Minutes', dependent_var])
    
    # Perform Pearson correlation test
    correlation, p_val = stats.pearsonr(df['Time_in_Minutes'], df[dependent_var])
    return correlation, p_val


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def model_training_and_evaluation(df, target_var, test_size=0.3):
    # Define features and target
    X = df[['Department', 'Year', 'Sector', 'Month', 'Time_in_Minutes', 'Safety_Hazard']]
    y = df[target_var]

    # Handle categorical data
    categorical_features = ['Department', 'Sector', 'Month', 'Safety_Hazard']
    numeric_features = ['Year', 'Time_in_Minutes']

    # Create transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create pipelines for KNN and Random Forest
    pipeline_knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train and evaluate KNN
    pipeline_knn.fit(X_train, y_train)
    y_pred_knn = pipeline_knn.predict(X_test)
    knn_results = {
        "classification_report": classification_report(y_test, y_pred_knn),
        "confusion_matrix": confusion_matrix(y_test, y_pred_knn),
        "accuracy": accuracy_score(y_test, y_pred_knn)
    }

    # Train and evaluate Random Forest
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    rf_results = {
        "classification_report": classification_report(y_test, y_pred_rf),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
        "accuracy": accuracy_score(y_test, y_pred_rf)
    }

    return knn_results, rf_results



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Load the uploaded CSV file into a pandas DataFrame
df_uploaded = pd.read_csv('selected_1000_rows_one_hot.csv')

# Display the first 5 rows of the DataFrame
df_uploaded.head()

def train_and_evaluate(df, col_dropped=None):
    # Split the data into features and target
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log the results with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_param("column_dropped", col_dropped)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
    
    return mse

def drop_and_evaluate(df):
    # Store the original columns
    original_columns = df.columns
    
    # Create a dictionary to store the MSE for each column dropped
    mse_results = {}
    
    # Start an MLflow run for drop and evaluate
    with mlflow.start_run(run_name="drop_and_evaluate"):
        # Iterate over each column (excluding the target column 'SalePrice')
        for col in original_columns:
            if col != 'SalePrice':
                # Drop the current column
                df_dropped = df.drop(columns=[col])
       
                # Train and evaluate the model with the column dropped
                mse = train_and_evaluate(df_dropped, col_dropped=col)
                # Store the resulting MSE
                mse_results[col] = mse   
                # Log the MSE for each column dropped
                mlflow.log_metric(f"mse_{col}", mse)
        print(mse_results)

if __name__ == '__main__':
    # Set the MLflow experiment name
    mlflow.set_experiment("LinearRegressionExperiment")
    
    # Train and evaluate the model on the uploaded dataset
    print('MSE with all features:', train_and_evaluate(df_uploaded))
    drop_and_evaluate(df_uploaded)

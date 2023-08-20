import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the uploaded CSV file into a pandas DataFrame
df_uploaded = pd.read_csv('selected_1000_rows_one_hot.csv')

# Display the first 5 rows of the DataFrame
df_uploaded.head()

def train_and_evaluate(df):
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
    
    return mse

def drop_and_evaluate(df):
    # Store the original columns
    original_columns = df.columns
    
    # Create a dictionary to store the MSE for each column dropped
    mse_results = {}
    
    # Iterate over each column (excluding the target column 'SalePrice')
    for col in original_columns:
        if col != 'SalePrice':
            # Drop the current column
            df_dropped = df.drop(columns=[col])
   
            # Train and evaluate the model with the column dropped
            mse = train_and_evaluate(df_dropped)
            # Store the resulting MSE
            mse_results[col] = mse   
    print(mse_results)

if __name__ == '__main__':
    # Train and evaluate the model on the uploaded dataset
    print('MSE with all features:', train_and_evaluate(df_uploaded))
    drop_and_evaluate(df_uploaded)

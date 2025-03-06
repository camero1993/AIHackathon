import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import os.path

def main():
    # Define file path
    data_file = "/Users/magnusgraham/Desktop/Cummins NSBE Innovation AI Impact-A-Thon/Data Set 1.xlsx"
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: File '{data_file}' not found.")
        return
    
    # Load data
    print("Loading data and training model...")
    df = pd.read_excel(data_file, engine="openpyxl")
    
    # Train model
    model, X, y, encoders = train_model(df)
    
    while True:
        # Display menu
        print("\n===== Demand Prediction Tool =====")
        print("1. Predict demand for a specific line number")
        print("2. Look up part and predict demand")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            predict_by_line(df, model, X, y)
        elif choice == "2":
            predict_by_part(df, model, X, y, encoders)
        elif choice == "3":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def train_model(df):
    """Train the decision tree model and return it along with prepared data"""
    # Select features (X) and target variable (y)
    X = df.iloc[:, 1:5]  # Columns 2,3,4,5 (part_number, product_category, Attribute_B, Attribute_C)
    y = df.iloc[:, 6]    # Column 6 (demand)
    
    # Store original column names
    original_columns = X.columns.tolist()
    
    # Encode categorical variables
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Train Decision Tree Regressor
    model = DecisionTreeRegressor(
        max_depth=34,
        splitter="best",
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. Mean Absolute Error: {mae}")
    
    return model, X, y, encoders

def predict_by_line(df, model, X, y):
    """Predict demand for a specific line number"""
    try:
        max_line = len(df) - 1
        line_number = int(input(f"Enter line number (0-{max_line}): "))
        
        if line_number < 0 or line_number > max_line:
            print(f"Error: Line number must be between 0 and {max_line}")
            return
        
        # Get the single sample
        single_sample = X.iloc[line_number:line_number+1]
        
        # Make prediction
        predicted_demand = model.predict(single_sample)[0]
        
        # Get actual values for display
        part_number = df.iloc[line_number, 1]  # Assuming part_number is column 2
        division = df.iloc[line_number, 2]     # Assuming division/product_category is column 3
        
        # Print the results
        print(f"\nResults for line {line_number}:")
        print(f"Part Number: {part_number}")
        print(f"Division/Category: {division}")
        print(f"Predicted demand: {predicted_demand:.2f}")
        
        # Compare with actual if available
        actual_demand = y.iloc[line_number]
        print(f"Actual demand: {actual_demand}")
        print(f"Difference: {abs(predicted_demand - actual_demand):.2f}")
        
    except ValueError:
        print("Error: Please enter a valid number.")
    except Exception as e:
        print(f"An error occurred: {e}")

def predict_by_part(df, model, X, y, encoders):
    """Look up a part and predict its demand using the ML model"""
    part_number = input("Enter Part Number: ").strip()
    division = input("Enter Division: ").strip().upper()
    
    # Convert necessary columns to string to avoid data type issues
    part_col = df.columns[1]  
    div_col = df.columns[5]   
    
    df_search = df.copy()
    df_search[part_col] = df_search[part_col].astype(str)
    df_search[div_col] = df_search[div_col].astype(str).str.upper()
    
    # Search for the matching rows
    matches = df_search[(df_search[part_col] == part_number) & 
                        (df_search[div_col] == division)]
    
    if not matches.empty:
        print(f"\nFound {len(matches)} matching records:")
        
        for i, row in matches.iterrows():
            # Get the corresponding row in the encoded X dataframe
            encoded_row = X.iloc[i:i+1]
            
            # Use the model to predict demand
            predicted_demand = model.predict(encoded_row)[0]
            
            # Get actual demand
            actual_demand = y.iloc[i]
            
            # Print details
            print(f"\nLine {i}:")
            print(f"Part Number: {row[part_col]}")
            print(f"Division: {row[div_col]}")
            
            # Print additional attributes if needed
            if len(df.columns) > 3:
                attr_b = df.iloc[i, 3]  # Assuming Attribute_B is column 4
                attr_c = df.iloc[i, 4]  # Assuming Attribute_C is column 5
                print(f"Additional Attributes: {attr_b}, {attr_c}")
            
            # Print demand predictions
            print(f"Predicted demand: {predicted_demand:.2f}")
            print(f"Actual demand: {actual_demand}")
            print(f"Difference: {abs(predicted_demand - actual_demand):.2f}")
            
    else:
        print("\nNo exact matches found.")
        
        # Suggest similar parts
        similar_parts = df_search[df_search[part_col].str.contains(part_number, case=False)]
        if not similar_parts.empty and len(similar_parts) <= 10:
            print("\nSimilar part numbers you might be looking for:")
            for part in similar_parts[part_col].unique()[:5]:
                print(f"- {part}")
                
        # Suggest divisions if part exists but in different division
        part_exists = df_search[df_search[part_col] == part_number]
        if not part_exists.empty:
            available_divisions = part_exists[div_col].unique()
            print(f"\nThis part number exists in the following divisions: {', '.join(available_divisions)}")

if __name__ == "__main__":
    main()
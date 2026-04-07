import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os

def preprocess_dataset(dataset_path):
    """Load and preprocess the dataset with comprehensive error handling"""
    try:
        # Load dataset with specific columns
        df = pd.read_csv(dataset_path)
        print("Successfully loaded dataset with shape:", df.shape)
        
        # Verify required columns exist
        required_columns = ['Weight', 'Height', 'BMI', 'Gender', 'Age', 'BMIcase', 'Exercise Recommendation Plan']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        df = df.dropna()
        print("After dropping NA values, shape:", df.shape)
        
        # Create BMI category from BMIcase
        df['BMI_Category'] = df['BMIcase'].apply(
            lambda x: str(x).strip().title().replace(' ', ''))
        
        # Clean recommendation plan column
        df['Exercise_Recommendation'] = df['Exercise Recommendation Plan'].astype(str).str.strip()
        
        # Encode categorical features with error handling
        encoders = {}
        categorical_cols = {
            'Gender': 'gender_encoder',
            'BMI_Category': 'bmi_encoder',
            'Exercise_Recommendation': 'exercise_encoder'
        }
        
        for col, encoder_name in categorical_cols.items():
            try:
                encoders[encoder_name] = LabelEncoder()
                df[col] = encoders[encoder_name].fit_transform(df[col])
                print(f"Successfully encoded {col}")
            except Exception as e:
                print(f"Error encoding {col}: {str(e)}")
                raise
        
        # Prepare features and target
        feature_cols = ['Weight', 'Height', 'Age', 'Gender']
        X = df[feature_cols]
        y = df['Exercise_Recommendation']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'encoders': encoders,
            'feature_names': feature_cols,
            'target_names': list(encoders['exercise_encoder'].classes_),
            'original_data_sample': df.head(3).to_dict()
        }
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        print("Sample of problematic data:")
        if 'df' in locals():
            print(df.head(3))
        raise

def save_artifacts(data, output_dir='model_artifacts'):
    """Save processed data with comprehensive checks"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encoders
        for name, encoder in data['encoders'].items():
            joblib.dump(encoder, f'{output_dir}/{name}.pkl')
        
        # Save other artifacts
        joblib.dump(data['feature_names'], f'{output_dir}/feature_names.pkl')
        joblib.dump(data['target_names'], f'{output_dir}/target_names.pkl')
        
        # Save numpy arrays
        np.save(f'{output_dir}/X_train.npy', data['X_train'])
        np.save(f'{output_dir}/y_train.npy', data['y_train'])
        np.save(f'{output_dir}/X_test.npy', data['X_test'])
        np.save(f'{output_dir}/y_test.npy', data['y_test'])
        
        # Save metadata
        with open(f'{output_dir}/preprocessing_report.txt', 'w') as f:
            f.write(f"Preprocessing Report\n{'='*20}\n")
            f.write(f"Original data sample:\n{data['original_data_sample']}\n")
            f.write(f"Feature names: {data['feature_names']}\n")
            f.write(f"Target classes: {data['target_names'][:10]}...\n")
            f.write(f"Train shape: {data['X_train'].shape}\n")
            f.write(f"Test shape: {data['X_test'].shape}\n")
        
        print(f"Successfully saved artifacts to {output_dir}")
        print(f"Feature names: {data['feature_names']}")
        print(f"First 5 target classes: {data['target_names'][:5]}")
        print(f"Train shape: {data['X_train'].shape}, Test shape: {data['X_test'].shape}")
        
    except Exception as e:
        print(f"Error saving artifacts: {str(e)}")
        raise

if __name__ == '__main__':
    dataset_path = 'data/final_dataset.csv'
    print(f"Starting processing of {dataset_path}")
    
    try:
        processed_data = preprocess_dataset(dataset_path)
        save_artifacts(processed_data)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify the CSV file exists at the specified path")
        print("2. Check the column names exactly match:")
        print("   - Weight, Height, BMI, Gender, Age, BMIcase, Exercise Recommendation Plan")
        print("3. Ensure there are no special characters or hidden spaces in column names")
        print("4. Check for missing values or inconsistent data types")
        print("5. Try opening the CSV in Excel/text editor to inspect the data")
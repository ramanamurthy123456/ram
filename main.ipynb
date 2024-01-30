from google.cloud import aiplatform

# Set your project ID and GCS bucket information
project = "canvas-abacus-411606"
location = "us-central1"
bucket_name = "amzu"
display_name = "your-model-display-name"
dataset_display_name = "your-dataset-display-name"
training_data_path = "gs://amzu/creditcard.csv"
set_target_cloumn = "Class"
# Initialize Vertex AI client
aiplatform.init(project=project, location=location)


# Create a tabular dataset with schema
tabular_dataset = aiplatform.TabularDataset.create(
    display_name=dataset_display_name,
    gcs_source=training_data_path,  # Specify the GCS bucket URL for your training data
)

# Create an AutoML tabular training job
automl_job = aiplatform.AutoMLTabularTrainingJob(
    display_name=display_name,
    optimization_prediction_type="classification",  # or "regression"
    optimization_objective = "maximize-au-roc"  # Use any supported option from the list
,  # or "minimize-rmse" for regression
)

# Start the training job and specify the dataset information
model = automl_job.run(
    dataset=tabular_dataset,
    target_column="Class",  # Specify your target column name here
    sync=True,
)

# Get the trained model
print("AutoML Model trained successfully:", model)


#=====================================deploy======================================================

# Deploy the trained model to an endpoint
endpoint = model.deploy(machine_type="n1-standard-8")  # Specify the machine type for deployment

# Wait for the deployment to complete
endpoint.wait()
#============================prediction==================================
# Prepare the input data for prediction
instances = [
     {
        "Time": "0",
        "V1": "-1.359807134",
        "V2": "-0.072781173",
        "V3": "2.536346738",
        "V4": "1.378155224",
        "V5": "-0.33832077",
        "V6": "0.462387778",
        "V7": "0.239598554",
        "V8": "0.098697901",
        "V9": "0.36378697",
        "V10": "0.090794172",
        "V11": "-0.551599533",
        "V12": "-0.617800856",
        "V13": "-0.991389847",
        "V14": "-0.311169354",
        "V15": "1.468176972",
        "V16": "-0.470400525",
        "V17": "0.207971242",
        "V18": "0.02579058",
        "V19": "0.40399296",
        "V20": "0.251412098",
        "V21": "-0.018306778",
        "V22": "0.277837576",
        "V23": "-0.11047391",
        "V24": "0.066928075",
        "V25": "0.128539358",
        "V26": "-0.189114844",
        "V27": "0.133558377",
        "V28": "-0.021053053",
        "Amount": "149.62",
    }
    
]

# Get predictions from the deployed model
predictions = endpoint.predict(instances)

# Print the predictions
print(predictions)
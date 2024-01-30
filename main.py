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
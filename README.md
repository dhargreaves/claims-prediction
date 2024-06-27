# Claims Predictions

#### What are the assumptions you have made for this service and why?
The assumptions are that the both the existing training data and the new data provided monthly will be available in the same table and can be pulled together for model retraining. That the data used for inference (new applications) is provided by the end of the working day each day and predictions need to be made for this data each day. The inference pipeline could be easily modified to run at the end of a 14 day cycle if required. The assumption is that batch inference with output written to a table would be sufficient given the needs of the business and the data size.

#### What considerations are there to ensure the business can leverage this service?
We would need to consider how the outputs from this model can be used. In the project developed, a batch inference pipeline has been setup to write to a table. This would allow BI teams to build a dashboard or for downstream systems to use as a source. Alternatively a live endpoint could be setup which could be used by a downstream system to pull predictions when required.


#### Which traditional teams within the business would you need to talk to and why?
The claims team would need to be involved at various touchpoints throughout the process to ensure the solution met their requirements. We would ideally also look for them to QA and perform some level of UAT on the finished output. As part of redevelopment it would also be worthwhile engaging with the claims team for any insight they might have that could lead to additional features being incorporated.
Compliance/Legal team should also be consulted to ensure the data is allowed to be used for analytical purposes.
Data engineering should be consulted to ensure the source data is provided and available in a way that will enable the pipelines to run correctly.
Any teams managing systems that will consume the output should be consulted to ensure the timing and formats of the data meet the requirements of the downstream system.
Some of this can be handled by developing data contracts as an operating principle between technical teams.

#### What is in and out of scope for your responsibility?
Any upstream ETL process to provide the training and inference data should be handled by data engineering and they should ensure the timing and quality of the data, however this should be at least checked by data science/ml engineering. 
Ensuring the performance of the model and the pipelines and then that model output is provided at agreed upon times and format are handled by the ML engineer. Any failures to the pipelines are also dealt with.
Once the outputs are provided in the agreed upon way then any issues with downstream systems should be handled by the teams responsible for those systems.
However, all teams should work together to ensure the full process works together and help with diagnostics in the case of any failures.


# Project Overview


1. Specification
2. Code Development
3. CI/CD
4. Additional Considerations

## Specification

To create productionised ML model with the following components:
    - Take notebook code and refactor into production grade code base
    - Setup CI/CD pipeline and deployment framework
    - Build inference pipeline to make predictions on newly available data provided daily
    - Update the model on a monthly basis with fresh training data provided


## Code Development

The code base was built using object orientated programming to allow components to be decoupled, enable easier testing and for components to be switched such as new models added or new data sources incorporated. 

The two main files are run_training_pipeline and run_inference_pipeline. Here these provide an interface for running the training and inference pipelines where different components such as the data loader, preprocessor, parameter tuner, model and evaluator can all be defined and switched for equivilant components if requried.

A strategy design pattern was used for the Model class to ensure any new models added would have the required methods to run in the training and inference pipelines. 

Both unit tests and integration tests have been added to test the code with some examples but not all have been coded in the project.

#### Note:
The model in the notebook used training data for validation as part of early stopping so this was modified to split out an additional validation dataset to use in this case. It's also worth noting that the data is heavily imbalanced and this hasn't been accounted for. In this instance if the imbalance in the data is a form of information (i.e. this is the probability of a claim being made) then this can be left as has been done here or the training can be modified to either undersample the training data or to provide the model with the imbalance ratio using the scale_pos_weight paramter,

## CI/CD

Most of the code development was done locally using Poetry to handle the local environment but then made the project a Databricks Asset Bundle and used Github Actions to setup the CI/CD pipeline. The Github repository has been setup with dev, release and main branches and the intention is to have 3 databricks environments for development, staging and production. 

The .github/workflows YAML files have been setup to run components of the CI/CD pipeline. The test.yml file triggers when a new commit is pushed to the dev branch. This performs code formatting using black, linting using pylint and unit testing using Pytest. Once a PR is raised on the main branch, the release.yml file installs the Databricks CLI and then deploys the Databricks Asset Bundle for the project and runs the training pipeline and integration tests. When the code is pushed to the main branch this triggers the Databricks Asset Bundle to be deployed to the production environment and scheduled jobs are setup for the training pipeline to run at midnight on the first of every month and the inference pipeline to run daily at 7pm.

## Additional Considerations

As this isn't a fully realised project yet, other functionality that would be considered as part of the project would be:

 #### Logging & Error Handling: 
 Ensure logging was setup to log any errors that occur through the pipeline with error codes logged along with the methods in which they occurred. In addition logging for the outcome of each method such as pre and post dataframe sizes should be logged to monitor any abnormal transformations that may occur.

 #### Inference Store & Monitoring
 In production both the inference data should be monitored for data drift and also predictions stored in a feature store and concept drift should be monitored for signs that the model should be redeveloped. Ordinarily this might be used to trigger a retrain but as the model is being trained monthly this is likely not needed.

 #### Code Review
A code review should take place at some point, likely between the point the dev code is being pushed to the release branch.

#### Feature Store
While the data processing steps in this instance are minor, a feature store should be considered for both training and inference data and the preprocessing stage seperated from the modelling stage. This could be handled by data science or data engineering. In addition if the data size increases and/or the preprocessing become more complex then it might be worth recoding this project in Spark.

#### Tests
Units test and integration tests components have been provided but not fully coded out. Unit tests should at least cover the primary methods in each class and issues that require fixes once the model is live should continually be incorporated into testing.

Once happy with offline testing of the model (the test dataset should only be used once), the model can be deployed as a shadow test where it makes predictions and these predictions are monitored but not used by the business to ensure the model performs as it did during the offline test. Once the business uses the output of the model, inital A/B tests could be performed against any existing process to determine business value generated by the model.

#### Champion Challenger Framework
A component has been provided for a champion / challenger module but not coded out. This would enable comparison of an existing model with a retrained or newly developed model and could be incorporated into the production deployment framework. This test should be run against primary evaluation metrics and can involve statistical A/B test.

#### Documentation
Docstrings should be provided for all classes and should be to a set style. Additional technical and design documents can be created to enable easier maintance or further development.

#### Further Training
The training pipeline uses separate hyperparameter tuning and training methods, mirroring the original notebook, however the random search model could be trained directly. If expanded experimentation is required when developing the model further, it may make sense to split the data out into k-fold stratified cross validation sets. Commonly 5-fold CV strategy has been shown to work well but is dependant on the size of the data.







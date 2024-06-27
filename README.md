# Claims Predictions



TO DO: Logging, monitoring, config, notes

Sections

Spec: 
Architecture: 2 phase ML pipeline
Code: Overview, what wasn't correct, design patterns, inference pipelines, tests, logging
Questions:
CI/CD: asset bundles, tests/linting etc, deployment, schedules
deployment: moniotoring, champion challenger
Other: Train RandomSearch directly, stratified k-fold for experimentation, spark for large data, tests, a/b test, feature store, logging, code review

1. Specification
2. Code Development
3. CI/CD
4. Deployment
5. Additional Functionality

## Specification

To create productionised ML model with the following components:
    - Take notebook code and refactor into production grade code base
    - Setup CI/CD pipeline and deployment framework
    - Build inference pipeline to make predictions on newly available data provided daily
    - Update the model on a monthly basis with fresh training data provided


## Code Development

The code base was built using object orientated programming to allow components to be decoupled, enable easier testing and for components to be switched such as new models added or new data sources incorporated. 

The two main files are run_training_pipeline and run_inference_pipeline. Here these provide and interface for running the training and inference pipelines where different components such as the data loader, preprocessor, parameter tuner, model and evaluator can all be defined and switched for equivilant components if requried.

A strategy design pattern was used for the Model class to ensure any new models added would have the required methods to run in the training and inference pipelines. 

Both unit tests and integration tests have been added to test the code with some examples but not all have been coded in the project.

#### Note:
The model in the notebook used training data for validation as part of early stopping so this was modified to split out an additional validation dataset to use in this case. It's also worth noting that the data is heavily imbalanced and this hasn't been accounted for. In this instance if the imbalance in the data is a form of information (i.e. this is the probability of a claim being made) then this can be left as has been done here or the training can be modified to either undersample the training data or to provide the model with the imbalance ratio using the scale_pos_weight paramter,

## CI/CD

Most of the code development was done locally using Poetry to handle the local environment but then made the project a Databricks Asset Bundle and used Github Actions to setup the CI/CD pipeline. The Github repository has been setup with dev, release and main branches and the intention is to have 3 databricks environments for development, staging and production. 

The .github/workflows YAML files have been setup to run components of the CI/CD pipeline. The test.yml file triggers when a new commit is pushed to the dev branch. This performs code formatting using black, linting using pylint and unit testing using Pytest. Once code is pushed to the release branch, the release.yml file installs the Databricks CLI and then deploys the Databricks Asset Bundle for the project and runs the training pipeline and integration tests. 





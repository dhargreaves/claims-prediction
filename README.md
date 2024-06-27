# claims-prediction



TO DO: Logging, monitoring, config, notes

Sections

Spec: 
Architecture: 2 phase ML pipeline
Code: Overview, what wasn't correct, design patterns, inference pipelines, tests, logging
Questions:
CI/CD: asset bundles, tests/linting etc, deployment, schedules
deployment: moniotoring, champion challenger
Other: Train RandomSearch directly, stratified k-fold for experimentation, spark for large data, tests, a/b test, feature store, logging

1. [Specification]
2. [Code Review]
3. [Code Design]
4. [CI/CD]
5. [Deployment]
6. [Additional Functionality]

## Specification:

To create productionised ML model with the following components:
    - Take notebook code and refactor into production grade code base
    - Setup CI/CD pipeline and deployment framework
    - Build inference pipeline to make predictions on newly available data provided daily
    - Update the model on a monthly basis with fresh training data provided


## Code Design

The code base was built using object orientated programming to allow components to be decoupled, enable easier testing and for components to be switched such as new models added or new data sources incorporated. 

The two main files are run_training_pipeline and run_inference_pipeline. Here these provide and interface for running the training and inference pipelines where different components such as the data loader, preprocessor, parameter tuner, model and evaluator can all be defined and switched for equivilant components if requried.

A strategy design pattern was used for the Model class to ensure any new models added would have the required methods to run in the training and inference pipelines.

The machine learning model was built 





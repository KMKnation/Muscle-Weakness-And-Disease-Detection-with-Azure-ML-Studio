# Muscle Weakness Detection using Microsoft Azure

In this project, we are going to train a model to detect progressive muscle weakness's status through multiple phenotype readings using Azure's AutoML and Hyperdrive process. After the training, we are going to deploy the best model and consuming it to check the working of endpoint.

## Dataset
### Overview

In this project, we are going to classify the progressive muscle weakness and disease with the use of HyperDrive and AutoML.

The dataset is custom data curated from one of the health institution.

The dataset contains 3 categories:
- Control
    
    Indicates whether muscle weakness is in control or not.
    
    
- SPG4
    
    **S pastic paraplegia 4 (SPG4)** is the most common type of hereditary spastic paraplegia (HSP) inherited in an autosomal dominant manner. Disease onset ranges from infancy to older adulthood. SPG4 is characterized by slowly progressive muscle weakness and spasticity (stiff or rigid muscles) in the lower half of the body. In rare cases, individuals may have a more complex form with seizures, ataxia, and dementia. SPG4 is caused by mutations in the SPAST gene. Severity of symptoms usually worsens over time, however some individuals remain mildly affected throughout their lives. Medications, such as antispastic drugs and physical therapy may aid in stretching spastic muscles and preventing contractures (fixed tightening of muscles) 
    
    Source : https://rarediseases.info.nih.gov/diseases/4925/spastic-paraplegia-4
    
    
- Disease
   
   Indicates that the muscle weakness is there due to some desease.


### Task
We are going to predict the status of muscle's weakness of the body. Human muscles are one of those part which gets started to weaken from the born. After some years, the muscles tissues starts to stiff. In  this case, we have to take more care of this muscles. Though, If we detect the status of muscle weakness in early stage then we might save many people from the illness cause by these muscles.

Here in this project, we are using HyperDrive and AutoML to train and to get the best model which is appropriate for the dataset.

### Access
The dataset is private. Hence to make the use of getting the data from the thirdparty, we used ngrok to host the data like following URL.

    path = "https://0547078f50ce.ngrok.io/data.csv"

and then using **TabularDatasetFactory** to retrive the data in workspace.

    ds = TabularDatasetFactory.from_delimited_files(path,
                                                validate=True,
                                                include_path=False,
                                                infer_column_types=True,
                                                separator=',',
                                                header=True,
                                                support_multi_line=False,
                                                empty_as_string=False)
## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

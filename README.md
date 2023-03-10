# ML-framework-on-Diamonds-dataset
This is an example repository to show how a basic ML framework looks like following OOP. The aim is to help anyone who is familiar with training ML models to transition from jupyter notebooks to OOP. 

Note that Jupyter notebooks are still better for Exploratory data anlysis and just building a first pass model. But this framework helps to create modular, easy to run code that can helpful to any project with minimal changes. 

Here are the steps in the current framework:  

1. split the given dataframe into train and test. This should be the first step to make sure nothing from test data is seen during training.  

2. Process training data and use encoding/scaling techniques if required. These encoders should be saved so that we can use it to process the test/inference data. 
   The data processor has 2 modes:
    i. Train mode - Fit the encoder and transform the data
    ii. Test mode - Load the encoder and transofrm the data. 

3. Training - Train the model. Also can use GridSearchCV for hyperpearameter tuning

4. Prediction - Get the model predictions on test dataset and save it as predictions.csv

5. Performance - output a classification report. can add any metric of interest and log them. 

Just run main.py to run through all the steps and post the precictions in the data folder. 

In general, the training and test predictions (inference) happen seperately. so can have a seperate training and an inference module going forward. 
Hope this helps! Please reach out in case of any questions.

   

# IoT Using LoRaWAN Uni Project

## System Overview

The main function of the system is to record and display various behavioral and movement patterns of a subject (person or animal). This is achieved by using an accelerometer and GPS attached to the subject which is then sent to Power BI via a LoRaWAN gateway and then displayed to the user using various analytical tools.  

The IoT Using LoRaWAN system aims to deliver the following outcomes:
* A group of sensors connected through LoRaWAN to a gateway
* A connection through the gateway directly to the IoT platform for sending and receiving the collected data
* An IoT platform set up for managing the data to and from the sensors
* A visualisation environment to illustrate the sensors activities
* A data processing algorithm to extract features and detect certain patterns in the logged data

The following high level system architecture diagram portrays the system:

![High Level Architecture Diagram](https://raw.githubusercontent.com/oreimu/IoT-Using-LoRaWAN-Uni-Project/master/images/figure2.jpg?token=AN6KF5Z4733KWPGO76YTT627O2MGS)

The files contained in this repository pertain to the functionality of the machine learning module that ran on the Raspberry Pi, including the training and testing of the model and the final program that ran on the pi `machineLearning.py`. The model was trained/generated in `trainModel.py` using large amounts of accelerometer data contained within a .csv file i.e. `Train_Huge.csv` then saved as `ML_Model5.h5`. This model is then used in `testModel.py` and tested against a smaller dataset of accelerometer data i.e. `test_accelerometer_data.csv` and generates a confusion matrix to show the accuracy of the model.

![Confusion Matrix](https://raw.githubusercontent.com/oreimu/IoT-Using-LoRaWAN-Uni-Project/master/images/Confusion%20Matrix.png?token=AN6KF5ZYO4MW5RF3RVCDRG27O2MF4)

## Methods Reference 

Below listed are all the Objects and their methods developed for this project for `machineLearning.py`: 

* **Function: add_feature(data)**
  This function is used to add features to the data. In the current method four features have been added which are Magnitude_vector, Axis1*Axis2, Axis2*Axis3, Axis1*Axis3. It has only one input parameter which is the whole raw data
* **Function: test_preprocess(time_steps,step)**
  This function is used to pre-process the test data in a way which the CNN model receive. The two input parameters time_steps and step both denotes the frequency of the data. For example: In the sample human data; the data is recorded with the frequency of 30Hz. So the value of time_steps and step will be 30.
* **Function: label_output(max_y_pred_test, step)**
  This function is used after the model has run it’s prediction and in the process compressed the predicted activity by a factor of 30. So this function reinflates the data so that it can be added back into the dataframe, by taking the max_y_pred_test as an argument and step as 1.
* **Function: Dataframe_trim(data, step_size)**
  This function is used to trim the pandas dataframe so that the predicted activity data can be added to the dataframe. As due to a limitation with pandas dataframe when adding columns to a dataframe the number of rows of the column being added and dataframe itself must match. So the method takes data as an input parameter and step_size as 30; matching the time_steps in test_preprocessing.

*Note: Due to time constraints for the project, we were unable to resolve an issue in saving data from the sensors on to the raspberry pi. Thus we were forced to send data from the network to our cloud platfrom, then pull data back from the platfrom to perform machine learning algorithms then send it back. Regardless, in order prove the proof of concept in time to the client it was necessary.*

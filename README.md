# Az-Ws-Designer-Classification - Diabetes Prediction Project ML Workflow
## 
https://www.coursera.org/learn/microsoft-azure-machine-learning-for-data-scientist/supplement/jbcrf/exercise-part-3-explore-and-prepare-data-using-azure-ml-designer
https://raw.githubusercontent.com/MicrosoftDocs/ml-basics/refs/heads/master/data/diabetes.csv


![image](https://github.com/user-attachments/assets/a716cc82-2096-4dfb-8116-c60317111115)
![image](https://github.com/user-attachments/assets/8446dd68-e1af-46e3-8585-df96c6bffcc7)
* Diabetes Training - Pipeline created in Designer
![image](https://github.com/user-attachments/assets/46cc9291-2d94-4b98-9d17-35a9bfa80f70)
![image](https://github.com/user-attachments/assets/400c0470-d1f8-40c3-a8ac-c6899cc6a86c)
![image](https://github.com/user-attachments/assets/29b6f373-e3f7-4202-8b6c-12331611db1d)
![image](https://github.com/user-attachments/assets/5c668649-fb64-40f9-8f70-7afba817588f)
![image](https://github.com/user-attachments/assets/6f6c1679-46c7-4f93-82a4-3b465fdf12a6)
![image](https://github.com/user-attachments/assets/37637e6c-f7ad-4f47-bd89-40f8a9d229bd)
![image](https://github.com/user-attachments/assets/4fc011c5-11e0-451c-af38-924f449337e5)
![image](https://github.com/user-attachments/assets/3e833654-fcef-4e28-be3a-e59c49c0f308)
![image](https://github.com/user-attachments/assets/6271cb4d-e881-41d2-959a-4cdd608b9700)
https://ml.azure.com/runs/e16d102a-07c0-4e8d-8ce7-14d933d653e2?wsid=/subscriptions/39eb117b-8d1f-4df9-af86-d4739857fabf/resourcegroups/rg-dp100-labs-designer/providers/Microsoft.MachineLearningServices/workspaces/mlw-dp100-labs-designer&tid=dd0f548c-6d16-4c4c-a83e-fbfa2ab3bf1a#/?graphId=ab9e41f3-8342-4599-981a-91e4891be084&label=Diabetes+Training&path=%2Fruns%2Fe16d102a-07c0-4e8d-8ce7-14d933d653e2&runId=e16d102a-07c0-4e8d-8ce7-14d933d653e2
![image](https://github.com/user-attachments/assets/87709cbd-431f-409d-b624-4b6ae0d82c7c)
![image](https://github.com/user-attachments/assets/afe64f81-4083-44b8-8b1e-8e4a36af3c99)
![image](https://github.com/user-attachments/assets/a3a674d2-1db0-4210-9676-2145d2989555)
![image](https://github.com/user-attachments/assets/bfa6f2fb-161a-497b-8282-2283ab1b8327)
Enter Data Manually module, containing the following CSV data, which includes feature values without labels for three new patient observations:

CSV Code Block
PatientID,Pregnancies,PlasmaGlucose,DiastolicBloodPressure,TricepsThickness,SerumInsulin,BMI,DiabetesPedigree,Age
1882185,9,104,51,7,24,27.36983156,1.350472047,43
1662484,6,73,61,35,24,18.74367404,1.074147566,75
1228510,4,115,50,29,243,34.69215364,0.741159926,59
--------------------------------------------------------------------------------------------------------------------------------------------
 Edit the Select Columns in Dataset module. Remove Diabetic from the Selected Columns.
 Delete the connection between the Score Model module and the Web Service Output.

Add an Execute Python Script module, replacing all of the default python script with the following code (which selects only the PatientID, Scored Labels and Scored Probabilities columns and renames them appropriately):
import pandas as pd
def azureml_main(dataframe1 = None, dataframe2 = None):
    scored_results = dataframe1[['Scored Labels', 'Scored Probabilities']]
    scored_results.rename(columns={'Scored Labels':'DiabetesPrediction',
                                'Scored Probabilities':'Probability'},
                        inplace=True)
    return scored_results
    -----------------------------------------------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/74e3a315-4acc-4305-b91d-e49acbe8cb3f)
![image](https://github.com/user-attachments/assets/28e33745-7fbb-409c-b6a6-670fc7538ea3)
![image](https://github.com/user-attachments/assets/a3a3f39f-5995-46a8-940e-4faf25c3f75c)
![image](https://github.com/user-attachments/assets/f618e28c-2d70-43e4-a574-3d076b4dd7cc)
![image](https://github.com/user-attachments/assets/87cb632e-79ca-4fc1-b683-9de464aa4efe)

![image](https://github.com/user-attachments/assets/80dae52e-09fc-4a34-9932-9d20d05036c0)

![image](https://github.com/user-attachments/assets/f601dedc-5929-4eaa-8313-c5cb3aec2666)
- Deploy
![image](https://github.com/user-attachments/assets/718bc6a6-ea48-4237-b587-15dd3149de62)

![image](https://github.com/user-attachments/assets/41a697f0-ccac-4fd3-b60d-7b678d0be168)
![image](https://github.com/user-attachments/assets/9a25a577-5918-4d1f-8c71-bcee44bd3801)
-----------------------------------------------

import urllib.request
import json

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = {}

body = str.encode(json.dumps(data))

url = 'http://7aaa67a6-2c2f-4114-b2dc-d93bd25c28c0.uksouth.azurecontainer.io/score'
# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
api_key = ''
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Accept': 'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))

----------------------------------------------------------------------------------------------------------------------------------
in advanced setting true for application insights
![image](https://github.com/user-attachments/assets/5fab4658-8776-4851-b252-1262b527d733)
![image](https://github.com/user-attachments/assets/3343fd36-ae88-4e14-bf14-b97d49539f5c)
![image](https://github.com/user-attachments/assets/daae624b-155d-4f48-8736-3826d2bb81b2)
![image](https://github.com/user-attachments/assets/2cfc908a-415e-4b54-8a20-188aa82a727c)
# Tested in Notebook SDK V2
----------------------------------------------------------------------------
import urllib.request
import json

# Request data goes here
data = {
    "Inputs": {
        "input1": [
            {
                "PatientID": 1882185,
                "Pregnancies": 9,
                "PlasmaGlucose": 104,
                "DiastolicBloodPressure": 51,
                "TricepsThickness": 7,
                "SerumInsulin": 24,
                "BMI": 27.36983156,
                "DiabetesPedigree": 1.3504720469999998,
                "Age": 43
            }
        ]
    },
    "GlobalParameters": {}
}

body = str.encode(json.dumps(data))

url = 'YOUR_ENDPOINT'
# Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
api_key = 'YOUR_KEY'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Accept': 'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result.decode("utf8"))
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
    print("Error Headers:")
    print(error.info())

    try:
        error_body = json.loads(error.read().decode("utf8", 'ignore'))
        print("\nError Body (JSON):")
        print(json.dumps(error_body, indent=4))
    except json.JSONDecodeError:
        print("\nError Body (Non-JSON):")
        print(error.read().decode("utf8", 'ignore'))


-------------------------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/c353229f-5b45-4a47-ac9b-49ef8c125a8e)
![image](https://github.com/user-attachments/assets/bb60e8ed-49de-4d65-9d07-ee8539507d23)
![image](https://github.com/user-attachments/assets/53e527c1-f3ee-4334-be84-e293f2fec7f0)
![image](https://github.com/user-attachments/assets/9cadba7f-4010-4c8e-babf-746a3aec16d6)

![image](https://github.com/user-attachments/assets/d633c221-5f69-49be-a933-29b629ab0b66)

----------------------------------------------------------------------------

<detail>

Step-by-Step Breakdown of the Modified Pipeline and the Role of the Python Script:

Enter Data Manually: This module is the starting point. It injects the CSV data you provided into the pipeline. This data represents the features (Pregnancies, PlasmaGlucose, etc.) for three new patients for whom we want to predict diabetes. Crucially, this data does not contain the "Diabetic" label because we are trying to predict it.

Select Columns in Dataset (Modified): You've edited this module to remove the "Diabetic" column. This makes sense because the scoring model expects input features only, not the target variable we are trying to predict.

Score Model: This module takes two inputs:

The output of the "Select Columns in Dataset" module, which now contains only the feature values for the new patients.
A trained machine learning model (likely a diabetes prediction model you trained previously in the Designer). The "Score Model" module applies the trained model to the input feature data and generates predictions. The output of this module typically includes:
Scored Labels: The predicted class label (e.g., 0 for no diabetes, 1 for diabetes).
Scored Probabilities: The probability of each class label (e.g., the probability of having diabetes).
Execute Python Script (Added): This is the crucial step where the provided Python script comes into play. It is connected to the output of the "Score Model" module. Here's what the script does:

Input: It receives the output of the "Score Model" as a Pandas DataFrame (dataframe1). This DataFrame contains columns like "Scored Labels" and "Scored Probabilities" (and potentially other metadata from the scoring process).
Selection: It selects only the "Scored Labels" and "Scored Probabilities" columns.
Renaming: It renames these columns to more user-friendly names:
"Scored Labels" is renamed to "DiabetesPrediction".
"Scored Probabilities" is renamed to "Probability".
Output: It returns a new Pandas DataFrame (scored_results) containing only the renamed "DiabetesPrediction" and "Probability" columns.
Web Service Output (Connection Deleted): You've deleted the direct connection from the "Score Model" to the "Web Service Output". This implies that you likely want to process or format the scoring results further before making them available as the output of a deployed web service. The "Execute Python Script" module now serves as an intermediate processing step.

When is the Execute Python Script Module Used?

In this modified pipeline, the Execute Python Script module is used immediately after the Score Model module. Its purpose is to post-process the raw scoring results generated by the model.

Elaboration on its Use and Advantages in this Step:

Data Transformation and Formatting: The "Score Model" output might contain more columns than you need for the final output of your web service or for further analysis. The Python script allows you to select and reshape the data to include only the most relevant information (the prediction and its probability).
Improved Readability and Clarity: The default column names ("Scored Labels", "Scored Probabilities") might not be as intuitive for someone consuming the web service output. Renaming them to "DiabetesPrediction" and "Probability" makes the results much clearer and easier to understand.
Preparation for Web Service Output (if reconnected later): If you were to reconnect the "Execute Python Script" module to the "Web Service Output", the output of the web service would be a clean and well-formatted JSON containing the "PatientID" (which is implicitly carried through), "DiabetesPrediction", and "Probability" for each patient.
Flexibility for Further Processing: This step provides a point where you could add more complex logic if needed. For example, you could implement thresholds for the probability to make a definitive "Diabetic" or "Not Diabetic" classification with a certain confidence level, or you could merge the prediction results back with the original patient IDs if those were carried through the pipeline.
In summary, the Execute Python Script module in this context acts as a data processing step that cleans, formats, and prepares the predictions from the Score Model before they are potentially outputted by a web service or used for other purposes. It enhances the usability and clarity of the prediction results.






</detail>




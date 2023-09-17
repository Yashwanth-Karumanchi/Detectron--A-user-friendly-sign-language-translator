# Detectron- A user friendly sign language translator

Heya fellas! 
There's a huge advancement in tech in the area of sign language translation, and yes, here's our take on it. 

This is a sign language translator that can generate speech output in any language that user wants for the signs of American Sign Language. Well, is it limited to ASL? NO. We have a fantastic gui that anyone can interact with to add any number of symbols from any set of signs. Then, train the model, and viola, there's your sign language translator.
For those tech savvy people, we also have a CLI version that you can access in the CLI codes folder. We used LSTM and MediaPipe to perform detection, with an accuracy of 99%.

The default detections that our model can make are provided in the defaultDetections.json file in the repo, and you can know what languages are supported from the supportedLanguages.json.
Our model can translate all english alphabets, 0-20 numbers and several gestures.

How to run it? Well, that's pretty easy for GUI.

  **PREREQUISITES**
1. Clone or download the repository
2. ENSURE PYTHON IS INSTALLED ON YOUR SYSTEM! If not the app will not open no matter how many times you click on it.
3. Double click on the install_requirements file to install all the dependencies. An alternate would be to open the command prompt in the directory where you have this repository, and execute 'pip install -r requirements.txt', and your dependencies are installed!
   
  **DATA MODIFICATIONS**
1.  To add data, double click on 'Modify Data' file in the App folder. A dialog box appears, using which you can modify the data.
2.  To create a new folder and add custom data, select a new folder. We recommend you to add to the data folder (selected by default) we have provided for addition of any new symbol.
3.  To replace or extend any existing symbol's data, choose replace data and extend data options option.

  **TRAINING**
1. To start the training, double click on 'Train the model' file in the App Folder. A dialog box with options for data path, epochs, patience and model name pops up.
2. If new data is added, specify the path to that folder. We recommend you to add data to the data folder we have provided and to use that as the path.
3. Unless familiar with the terms epochs and patience, do not change them. Prefer using the default numbers. Our model detects 62 signs and gestures. If the signs or gestures cross 75 in data folder, please increase the epochs to 700
4. Provide the name you wish to have for the saved model in model name field.
5. Click on start training and there it is! Your model is built and is stored at runs folder in main directory where you have the repository downloaded! (Usually takes time. Donot panic if slow)
    

  **DETECTION** 
1. To start detection, navigate to App folder, and double click on 'Detect' file to start detection. A dialog box pops up! The default values are set here, which the user can change using our GUI.
2. Choose the folder where your data is present, else leave it on default, choose any model you have trained and select the language for speech generation.
3. The sign language recognition starts! You can press Q to generate speech output and E to exit. If any wrong detection's take place, use BACKSPACE to erase it. If a backspace is clicked by mistake, click on R to undo deletion.
4. The detections including the speech and text files are stored in Detections folder on the main folder, where you have downloaded the repository.

FOR MORE TECH PERSONS THAT WANT TO PLAY AROUND USING CLI:

1. You can navigate to CLI codes folder, where you can execute the file on Command Prompt. 
2. For modification of data, you can execute the command 'python create.py' to interact with the data. Again, default is data folder that we provided
3. For training purposes, you can execute "python train.py --epochs 600 --data ../data --patience 100 --model exp". Of course, those values can be changed to specify number of epochs to run training for, data       folder path, patience for early stopping of the model, and name for the model to be saved with.
4. For detection purposes, you can execute "python detect.py --data ../data --model ../models/model.h5 --lang english". Of course, those values can be changed to specify the data folder path, model path to be       used and the language to generate speech.

--ALSO...this is just an experimental approach we have achieved, so please ensure to have good lighting and webcam for detection purposes--   

And there's that. This is our take on Sign Language Translation for people in needs. For people who are interested in this domain, feel free to use our work for reference or even develop on it. Thank you.

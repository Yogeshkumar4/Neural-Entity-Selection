# Neural-Entity-Selection


To train the Neural entity selection model, set the train flag in the model.py file to "True". 
The Model takes in NES_Dataset.txt as input which is present in the Data folder. 
To create your own data set, use the NES_dataset.py file present in Data Preprocessing directory which takes in the src-featurised file and answer files(it's right now written for train instance where len(src-featurised) != len(answer_dev)).

Setup tensorflow using conda, instructions are "https://www.tensorflow.org/install/install_linux#InstallingAnaconda"
Run python model.py and the train files will be generated.


For encoding answers in test files, set the train flag to "False"
Then generate data set using NES_dataset.py and give in as input src-feature.txt and raw sentences which are generated using util.py in Data proprocessing directory

# Anomaly_Detection

This is a hybrid model for Intrusion Detection which uses both Anomaly Detection models and Misuse Detection models.
A few intructions to be followed to set up an execution environment:

Preferred execution environment: IDE like Spyder or Jupyter. R code can be executed in RStudio

1] pydot and graphviz libraries need to be installed for visualisation of the decision trees in the Random Forest Model. Either pip install or conda install can be used to do the same.

2] For Windows, graphviz needs to be added to the PATH environment variable using the command:
os.environ[\"PATH\"] += os.pathsep + (path to graphviz)

3] When running in Spyder or a single cell in Jupyter, multiple graphs cannot be visualised together. We request you to comment out the others and view one at a time.

4] Install ann_visualizer for visualizing the Neural Network using the command:
pip3 install ann_visualizer

5] On MacOS, the code_nn.py file might throw kernel errors. To resolve this install 'nomkl' using: conda install nomkl. If that doesn't work then set environment variable KMP_DUPLICATE_LIB_OK to true using: os.environ['KMP_DUPLICATE_LIB_OK']='True' 

6] Keras with a TensorFlow base is required to run the Neural Network Model.

7] All tests were carried out in Anaconda on a MacOS. So there are chances of discrepencies.

8] Execute the file names IDS.py. The other models are called as  a part of this file. This file is the culmination point. The entire model takes approxiamtely 40 minutes to execute.

9] There is another file of Misuse Clustering called UnknownAttacksDetection.py which will run on a standalone basis having a classification for attacks not known in the training stage. This file needs to be executed seperately.

10] Seed has not been set to reproduce the exact same results. The average results obtained will still be the same although induvidual accuracies might vary. Overall we have not noticed any huge drop in accuracy. However the results might not be exactly the same as the documented results.

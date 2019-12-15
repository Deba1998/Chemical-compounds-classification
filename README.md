# Chemical-compounds-classification
Here with create a classification model of chemical compound using Multi layer perception(ANN) 
First we preprocess the data and remove the variable names and form X matrix with independent variables
We created the y matrix with dependent variables
Then we divide the data into training and test set in ratio 80:20
Then we apply feature scaling on matrix x to scale the variables of x matrix
Then we create a sequential model
Then we add our input and hidden layer which consists of 166 nodes as our matrix X contains 166 different features
Then we similarly add a extra input layer 
We use "relu" activation function in order to achieve non linearity
Then we add our output layer and use sigmoid function as this classification is a type of binary classification
Then we compile the model using adam optimizer and set the loss as binarycrossentropy as it is a binary classification problem
We use accuracy metrics to obtain accuracy in differnet epochs
Then we train the model using 50 epochs and batch size of 50 and save it to "hist1" for graph plotting
Then we predicts the test set values and use to make  confusion matrix with ytest and y pred 
Then we predict the precision score recallscore and F1 score
Then we again run our classifier in test set and store it in variable "hist2 " for graph plotting
Then we show the mean accuracy and loss of our validation data  tha tis test set
Then we plot the graphs of epochs vs loss and epochs vs accuracies by taking only first 9 epochs 

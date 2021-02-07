# FmPM
Identify particles in microscopy images as miroplastic or non-microplastic. 

Use Case Information
* User: Microplastic scientist
* Job/role: researcher studying microplastics, specifically microplastis in oceans/bodies of water
* Case 1: Classification of an image
  * User: a microplastic scientist with an image or set of images that she/he would like to classify as microplastic or non-microplastic
  * Verification that input data is an image (.bmp only?)
  * Check/raise error/treat differently if there are multiple particles
  * System categorizes particle color, shape, and size
  * System classifies particle as microplastic or non-microplastic
  * Alow to choose a saved model that has been previously trained 
  * Stretch case: system classifies type of microplastic 
* Case 2: Retraining on the model 
 * User has chemically analyzed data that she/he wants to use to retrain the model (e.g. images from a different lab)
 * User likely does not have a strong coding background
 * Allow the user to retrain the model on the images and data that they have
 * Allow user to choose training/validation set sizes 
 * Allow user to easily load the data- format of training data (chemical analysis info)? 
 * Save model 
* Case 3: User only wants size and/or shape information
 * User inputs images that he/she wants information only on the size and/or shape of the particle
 * Skip classification steps
 * Output determined size and/or shape information only 
* Case 4: Hyperparameter tuning
 * User has some coding background and would like to try tuning the hyperparameters
 * Allow the user to change hyperparameters (e.g. number of layers, neurons, activation functions)
 * Prompt the user to retrain the model after changing hyperparameter(s)
 * Output accuracy after training by testing on the validation set
 * Save the model 

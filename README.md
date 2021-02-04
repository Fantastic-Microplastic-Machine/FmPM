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
  * System classifies type of microplastic? 
* Case 2:
 * User has chemically analyzed data that she/he wants to use to retrain the model (e.g. images from a different lab)
 * User likely does not have a strong coding background
 * Allow the user to retrain the model on the images and data that they have
 * Allow user to choose training/validation set sizes 
 * Allow user to easily load the data- format of training data (chemical analysis info)? 

* Classification method
* Model generation

* Error messages: image doesn't have a particle or there is more than one particle on it. 
* Stretch cases: multiple particles/image? Particles obtained from soils? Classifying type of plastic?

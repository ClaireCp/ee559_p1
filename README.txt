The project code is structured as follows, the root folder contains:

    - a folder per framework: each framework is encompassed in its own folder, namely channelimagesModels, weightssharingModels and auxiliaryLossesModels. Each folder contains 2 modules, a first one defining the different Neural Network architectures that were tested, and another to define the relevant train_model(), test_model() and multiple_training_runs() functions related to the framework.
    
    - a generic_helpers.py module, which contains a write_to_csv() function, which is used to write the results obtained for the _2channels1image, weight_sharing and auxiliary_losses frameworks, as well as a compute_properties() function to compute the mean and standard deviation of the list passed as parameter
    
    - a data folder which contains the data and the dlc_practical_prologue.py module to load the data
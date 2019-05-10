import sys
sys.path.insert(0, "../")

from _2channels1image import train_model_2C
from _2channels1image import test_model_2C

""" FRAMEWORK FOR WEIGHTSHARING """
""" As for the _2channels1image framework, the network is trained to directly predict if the first digit is les or equal to the second. The network is a binary classifier and so we use BCEWithLogitsLoss. Thus, the particularity of this framework only comes from the architecture of the neural network itself. The training and testing functions are thus the same as for the _2channels1image framework. """

train_model_ws = train_model_2C
test_model_ws = test_model_2C

title_ws = 'Plot for 10 runs with weight_sharing framework (binary cross-entropy loss) and model = {}, \n accuracy obtained during training (model in eval mode) on the training set, and on the test set'
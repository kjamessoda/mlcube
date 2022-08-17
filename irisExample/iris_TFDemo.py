import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

#This function will convert the species into integers
def convertSpeciesToInteger(species):
    if species == 'Iris-versicolor':
        speciesInt = 0
    elif species == 'Iris-virginica':
        speciesInt = 1
    elif species == 'Iris-setosa':
        speciesInt = 2
    else:
        raise ValueError("convertSpeciesToInteger encountered an unknown species.")
    return speciesInt

#Initially import the data via pandas and split it into test and training
irisData = pd.read_csv("IRIS.csv").sample(frac = 1.,random_state = 304699040)

input_train    = irisData.iloc[:100,0:4]
output_train   = [convertSpeciesToInteger(i) for i in irisData.iloc[:100,4]]
trainingIrises = tf.data.Dataset.from_tensors((input_train,output_train)).shuffle(1000).batch(2)

input_test    = irisData.iloc[100:,0:4]
output_test   = [convertSpeciesToInteger(i) for i in irisData.iloc[100:,4]]
testIrises = tf.data.Dataset.from_tensors((input_test,output_test)).shuffle(1000).batch(2)


#Define the ANN architecture
class irisNetwork(Model):
    def __init__(self):
        super(irisNetwork,self).__init__()
        self.inputLayer  = Dense(4)
        self.hidden      = Dense(50,activation = 'relu')
        self.outputLayer = Dense(3)

    def call(self,x):
        x = self.inputLayer(x)
        x = self.hidden(x)
        return self.outputLayer(x)

#Initialize the model
liveNetwork = irisNetwork()

#Establish the loss function, the optimizer, and the metrics (really, this is taken from the Tutorial)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

meanLoss_training = tf.keras.metrics.Mean(name = 'meanLoss_training')
accuracy_training = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy_training')

meanLoss_test = tf.keras.metrics.Mean(name = 'meanLoss_test')
accuracy_test = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy_test')

#Define what should happen in one epoch
@tf.function
def trainingRound(inputs,outputs):
    with tf.GradientTape() as tape:
        predictedSpecies = liveNetwork(inputs,training = True)
        currentLoss = loss_func(outputs,predictedSpecies)
    grads = tape.gradient(currentLoss,liveNetwork.trainable_variables)
    optimizer.apply_gradients(zip(grads,liveNetwork.trainable_variables))

    meanLoss_training(currentLoss)
    accuracy_training(outputs,predictedSpecies)

#Define what should happen when we test the network at the end of an epoch
@tf.function
def testingRound(inputs,outputs):
    predictedSpecies = liveNetwork(inputs,training = False)
    currentLoss = loss_func(outputs,predictedSpecies)

    meanLoss_test(currentLoss)
    accuracy_test(outputs,predictedSpecies)

EPOCHS = 10

for epoch in range(EPOCHS):
    meanLoss_training.reset_states()
    accuracy_training.reset_states()
    meanLoss_test.reset_states()
    accuracy_test.reset_states()

    for flower, species in trainingIrises:
        trainingRound(flower,species)

    for flower, species in testIrises:
        testingRound(flower,species)

    #Report how this epoch went (taken from the Tutorial)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {meanLoss_training.result()}, '
        f'Accuracy: {accuracy_training.result() * 100}, '
        f'Test Loss: {meanLoss_test.result()}, '
        f'Test Accuracy: {accuracy_test.result() * 100}'
      )

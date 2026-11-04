#import os
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# THIS IS TO SELECT GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=str(3) # CAN BE 0,1,2, or 3

# THIS IS THE TRAIN AND TEST DATASET
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# THIS DEFINES THE NEURAL NETWORK
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# THIS COMPILES THE NEURAL NETWORK
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# THIS TRAINS THE NEURAL NETWORK ON THE TRAIN DATASET
model.fit(x_train, y_train, batch_size=32,epochs=10)

# THIS EVALUATES THE NEURAL NETWORK ON THE TEST DATASET
score=model.evaluate(x_test, y_test)
print(score[0], score[1])

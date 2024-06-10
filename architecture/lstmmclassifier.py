#data preprocessing
import pandas as pd
import numpy as np
import time as t

df = pd.read_csv("merger.csv")
if df.isna().any().any():
          df.isna().sum()
          print("removing the num values")
          t.sleep(3)
          df.dropna()

df["binary"]=[
    1 if x < len(df["Time (s)"]) else 1 for x in range(0,len(df["Time (s)"]))
]
y_train= df["binary"]
x_train=df.drop("binary",axis=1)


dt = pd.read_csv("data.csv")
dt["bin"] = [
    1 if x < len(dt["Time (s)"]) else 1 for x in range(0,len(dt["Time (s)"]))
]
y=dt["bin"]
dt=dt.drop("bin",axis=1)

x=np.array(dt)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_test,x_val,y_test,y_val = train_test_split(x,y,test_size=0.2,shuffle=False,random_state=False)

x_train = np.array(x_train)
y_train = np.array(y_train)
print(y_train.shape)
print(x_train.shape)

def create(x,t):
  d=[]
  for i in range(0,x.shape[0]):
     a = x[i:i+t]
     d.append(a)
  return   np.array(d)

x_train = create(x_train,1)
#x_train.reshape(x_train.shape[0],1,x_train.shape[1])
y_train=y_train.reshape(y_train.shape[0],1)
print(y_train.shape)
print(x_train.shape)
x_val = create(x_val,1)
x_test = create(x_test,1)

y_val = y_val.reshape(y_val.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
print(y_val.shape)
print(y_test.shape)

#model architecture 
import tensorflow as tf
import keras
from keras import layers
from keras import models

model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True,input_shape=(1,7)),merge_mode="ave"))
model.add(layers.Bidirectional(layers.LSTM(64,return_sequences=False ),merge_mode="ave"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation = "relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy","Precision","Recall"])
model.fit(x_train,y_train,epochs=10,validation_data=(x_val,y_val))
p=model.predict(x_test)

#saving the model  for furthe use we can use three different save types
model.save("savemodel/my_model")

#optimization of saved model to degrade the model size with convertion of saved model to tflite model
import os
print(os.path.getsize("savemodel/my_model")/1024) #in mb's
co= tf.lite.TFLiteConverter.from_saved_model("savemodel/my_model")
co.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite built-in ops.
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow Select ops.
]

# Disable experimental lowering of tensor list ops
co._experimental_lower_tensor_list_ops = False
co.optimizations=[tf.lite.Optimize.DEFAULT ]
tmodel= co.convert()

tflite="tflite"
open(tflite,"wb").write(tmodel)
size=os.path.getsize(tflite)
print(round(size/(1024)))

#converting the tflite to .h file
!echo "const unsigned char model[]= {" > tml.h
!cat tflite | xxd -i >> tml.h
!echo "};" >> tml.h
size = os.path.getsize("tml.h")
print((size/(1024)))


#for testing the tflite converted model
interpreter = tf.lite.Interpreter(tflite)
interpreter.allocate_tensors()
input_data = np.array([[0,-2.71905,9.603001,.00705,9.980527,89.959526,105.80925]],dtype=np.float32)
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)
input_data = np.reshape(input_data, input_shape)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Get output details
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("TensorFlow Lite model output:", output_data)


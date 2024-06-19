import numpy as np
import pandas as pd
import time as t

import tensorflow as tf
import keras
from keras import layers
from keras import models

#model architecture
model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True,input_shape=(1,7)),merge_mode="ave"))
model.add(layers.Bidirectional(layers.LSTM(64,return_sequences=False ),merge_mode="ave"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation = "relu"))
model.add(layers.Dense(2,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy","Precision","Recall"])

num =2

f="2"
m="3"
mi="1"
fi="0"

d=[0 for _ in range(num)]
d1=[0 for _ in range(num)]

for  i in range(0,numberofpersons-1):

 input = "loopdata/" + str(i) + ".csv"
 #data preprocessing for loop the model.fit
 print(input + "*****************************" )
 df = pd.read_csv(input)
 if df.isna().any().any():
          df.isna().sum()
          print("removing the num values")
          df.dropna()

 def create(x,t):
   d=[]
   for i in range(0,x.shape[0]-t-1):
      a = x[i:i+t]
      d.append(a)
   return   np.array(d)


 for i in range(0,num):
    d[i]=str(i)

 print("values in d...")
 print(d)

 #dataset with one hot encoding
 print("working for 0 and 1 assignment ")

 for col in d:

  if col in df.columns:
    df.loc[:len((df["Time (s)"]))-1,col]=1
    print(df.head(1))


  else :
     lst = df.columns[-1]
     if int(lst)>int(col):
       dfp=df.pop(lst)
     df.loc[:len(df["Time (s)"]-1),col]=0
     if int(lst)>int(col):
      df[lst]=dfp
     print(df.head(1))


 print("after working")
 print(df.head(1))

 y=df.iloc[:,-num:]
 de = df.columns[-num:]
 x_train=df.drop(de,axis=1)
 x_train = create(x_train,1)
 y_train = y.head(x_train.shape[0])
 y_train = np.array(y_train)




 dt = pd.read_csv("81.csv")
 #dt["bin"] = [
 #   1 if x < len(dt["Time (s)"]) else 1 for x in range(0,len(dt["Time (s)"]))
 #]
 print("started for validation")
 for i in range(num):
  d1[i]=str(i)
 print(d1)


 for col in d1:
  if col in dt.columns:
    dt.loc[:len(dt["Time (s)"])-1,col]=1
    print(dt.head(1))

  else :
    ls = dt.columns[-1]
    if int(ls) > int(col):
      dtp =dt.pop(ls)
    dt.loc[:len(dt["Time (s)"])-1,col] =0
    if int(lst)>int(col):
      df[lst]=dtp
    print(dt.head(1))


 print("after working for validation")
 y= dt.iloc[:,-num:]
 dee = dt.columns[-num:]
 x=dt.drop(dee,axis=1)
 x = create(x,1)
 y = y.head(x.shape[0])
 y = np.array(y)


 print(x_train.shape)
 print(y_train.shape)

 print(x.shape)
 print(y.shape)

 #from sklearn.model_selection import train_test_split
 #x_test,x_val,y_test,y_val = train_test_split(x,y,test_size=0.2,shuffle=False,random_state=False)

 #tf.one_hot(y_train,1)
 #tf.one_hot(y,1)

 #tf.one_hot(y_test,1)

 print(y_train)

 #x_train.reshape(x_train.shape[0],1,x_train.shape[1])
 #y_train = np.mean(y_train, axis=1).reshape(-1, 1)
 model.fit(x_train,y_train,epochs=5,validation_data=(x,y))
 #model.predict(x_test)

#creating the data for testing using test data
dc = pd.read_csv("9.csv")
#8 male 9 female
tc = dc.shape[0]-2
if dc.isna().any().any():
  dc.dropna()
def creat(x,t):
   d=[]
   for i in range(0,x.shape[0]-t-1):
      a = x[i:i+t]
      d.append(a)
   return   np.array(d)

x_t = creat(dc,tc)
print(x_t.shape)



#predicitng the output of the test data
#x_t is test data which we use for testing the model accuracy
model.predict(x_t)


#saving the model got conversio
model.save("savemodel/my_model")


#checking the size of the model
import os
print(os.path.getsize("savemodel/my_model")/1024) #in mbs


#converting the model to tflite using tflite converter
co=tf.lite.TFLIteConverter.from_saved_model("savemodel/my_model")
co.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite built-in ops.
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow Select ops.
]

# Disable experimental lowering of tensor list ops
co._experimental_lower_tensor_list_ops = False
#optimization of the weights then converting to tflite
co.optimizations=[tf.lite.Optimize.DEFAULT ]
tmodel= co.convert()


#writing the contents in tflite.tflite to tmodel (tmodel which we got after we converted the model to tflite and also optimzatio )
tflite="tflite.tflite"
open(tflite,"wb").write(tmodel)
size=os.path.getsize(tflite)
print(size/1024)


#now we are converting the converted contents to c array for forming .h file for arduinolibraries
!echo "const unsigned char model[]= {" > tml.h
!cat tflit.tflite | xxd -i >> tml.h
!echo "};" >> tml.h
size = os.path.getsize("tml.h")
print((size/(1024)))


#this part is for testing the c array file or .h file which we just created

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





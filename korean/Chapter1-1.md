# 1-1 예제: 구조화된 데이터를 위한 모델링 순서 


### 1. 데이터 준비 


타이타닉 데이터 셋의 목적은 타이타닉이 빙산에 충돌한 이후 주어진 탑승객 정보를 이용해, 어떤 탑승자가 살아남을 수 있을지를 예측하는 것입니다. 
일반적으로 Pandas 라이브러리의 DataFrame을 사용하여 데이터를 전처리합니다.

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import models,layers

dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/test.csv')
dftrain_raw.head(10)
```

![](../data/1-1-数据集展示.jpg)


각 항목에 대한 설명：

* Survived: 0 : 사망, 1 : 생존 [y labels]
* Pclass: 탑승권의 등급, 1,2,3 값을 가짐 [converting to one-hot encoding]
* Name: 승객 이름 [discarded]
* Sex: 승객의 성별 [converting to bool type]
* Age: 승객의 나이 (일부 데이터는 누락됨) [numerical feature, should add "Whether age is missing" as auxiliary feature]
* SibSp: 승객의 형제, 자매, 배우자 수 (interger) [numerical feature]
* Parch: 각 승객의 부모와 자녀 수 (interger) [numerical feature]
* Ticket: 탑승권 번호 (string) [discarded]
* Fare: 각 승객의 탑승권 가격 (float, between 0 to 500) [numerical feature]
* Cabin: 각 승객의 선실 위치 (일부 데이터 누락됨) [should add "Whether cabin is missing" as auxiliary feature]
* Embarked: 각 승객이 탑승한 항구, S, C, Q 값을 가짐 ( 일부 누락됨 ) [converting to one-hot encoding, four dimensions, S,C,Q,nan]


pandas의 라이브러리를 이용하여 탐색적 데이터 분석을 위한 시각화.

생존자 분포:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'png'
ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',
     figsize = (12,8),fontsize=15,rot = 0)
ax.set_ylabel('Counts',fontsize = 15)
ax.set_xlabel('Survived',fontsize = 15)
plt.show()
```

![](../data/1-1-Label分布.jpg)


나이 분포:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'png'
ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
                    figsize = (12,8),fontsize=15)

ax.set_ylabel('Frequency',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()
```

![](../data/1-1-年龄分布.jpg)


나이와 생존과의 상관 관계:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'png'
ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
ax.legend(['Survived==0','Survived==1'],fontsize = 12)
ax.set_ylabel('Density',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()
```

![](../data/1-1-年龄相关性.jpg)


다음은 전처리를 위한 코드 입니다:

```python
def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )

```

```
x_train.shape = (712, 15)
x_test.shape = (179, 15)
```

```python

```

### 2. 모델 정의


일반적으로 Keras에서 사용하는 모델링 방법은 3가지가 있습니다. : 'Sequential'함수를 이용한 순차 모델링, API를 이용한 임의 모델링, 기본 클래스 'Model'을 상속하여 커스텀 모델링.

여기서는 가장 간단한 'Sequential'함수를 이용한 순차 모델링 방법을 사용합니다.

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 20)                320       
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11        
=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0
_________________________________________________________________
```


### 3. 모델 학습 


모델을 학습하는데는 일반적으로 3가지 방법이 있습니다. : fit 함수를 사용하는 것, train_in_batch 함수를 이용하는 것, 그리고 사용자가 정의한 반복 학습. 여기서는 가장 간단한 fit 함수를 이용하는 방법을 사용합니다. 

```python
# Use binary cross entropy loss function for binary classification
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2 #Split part of the training data for validation
                   )
```

```
Train on 569 samples, validate on 143 samples
Epoch 1/30
569/569 [==============================] - 1s 2ms/sample - loss: 3.5841 - AUC: 0.4079 - val_loss: 3.4429 - val_AUC: 0.4129
Epoch 2/30
569/569 [==============================] - 0s 102us/sample - loss: 2.6093 - AUC: 0.3967 - val_loss: 2.4886 - val_AUC: 0.4139
Epoch 3/30
569/569 [==============================] - 0s 68us/sample - loss: 1.8375 - AUC: 0.4003 - val_loss: 1.7383 - val_AUC: 0.4223
Epoch 4/30
569/569 [==============================] - 0s 83us/sample - loss: 1.2545 - AUC: 0.4390 - val_loss: 1.1936 - val_AUC: 0.4765
Epoch 5/30
569/569 [==============================] - ETA: 0s - loss: 1.4435 - AUC: 0.375 - 0s 90us/sample - loss: 0.9141 - AUC: 0.5192 - val_loss: 0.8274 - val_AUC: 0.5584
Epoch 6/30
569/569 [==============================] - 0s 110us/sample - loss: 0.7052 - AUC: 0.6290 - val_loss: 0.6596 - val_AUC: 0.6880
Epoch 7/30
569/569 [==============================] - 0s 90us/sample - loss: 0.6410 - AUC: 0.7086 - val_loss: 0.6519 - val_AUC: 0.6845
Epoch 8/30
569/569 [==============================] - 0s 93us/sample - loss: 0.6246 - AUC: 0.7080 - val_loss: 0.6480 - val_AUC: 0.6846
Epoch 9/30
569/569 [==============================] - 0s 73us/sample - loss: 0.6088 - AUC: 0.7113 - val_loss: 0.6497 - val_AUC: 0.6838
Epoch 10/30
569/569 [==============================] - 0s 79us/sample - loss: 0.6051 - AUC: 0.7117 - val_loss: 0.6454 - val_AUC: 0.6873
Epoch 11/30
569/569 [==============================] - 0s 96us/sample - loss: 0.5972 - AUC: 0.7218 - val_loss: 0.6369 - val_AUC: 0.6888
Epoch 12/30
569/569 [==============================] - 0s 92us/sample - loss: 0.5918 - AUC: 0.7294 - val_loss: 0.6330 - val_AUC: 0.6908
Epoch 13/30
569/569 [==============================] - 0s 75us/sample - loss: 0.5864 - AUC: 0.7363 - val_loss: 0.6281 - val_AUC: 0.6948
Epoch 14/30
569/569 [==============================] - 0s 104us/sample - loss: 0.5832 - AUC: 0.7426 - val_loss: 0.6240 - val_AUC: 0.7030
Epoch 15/30
569/569 [==============================] - 0s 74us/sample - loss: 0.5777 - AUC: 0.7507 - val_loss: 0.6200 - val_AUC: 0.7066
Epoch 16/30
569/569 [==============================] - 0s 79us/sample - loss: 0.5726 - AUC: 0.7569 - val_loss: 0.6155 - val_AUC: 0.7132
Epoch 17/30
569/569 [==============================] - 0s 99us/sample - loss: 0.5674 - AUC: 0.7643 - val_loss: 0.6070 - val_AUC: 0.7255
Epoch 18/30
569/569 [==============================] - 0s 97us/sample - loss: 0.5631 - AUC: 0.7721 - val_loss: 0.6061 - val_AUC: 0.7305
Epoch 19/30
569/569 [==============================] - 0s 73us/sample - loss: 0.5580 - AUC: 0.7792 - val_loss: 0.6027 - val_AUC: 0.7332
Epoch 20/30
569/569 [==============================] - 0s 85us/sample - loss: 0.5533 - AUC: 0.7861 - val_loss: 0.5997 - val_AUC: 0.7366
Epoch 21/30
569/569 [==============================] - 0s 87us/sample - loss: 0.5497 - AUC: 0.7926 - val_loss: 0.5961 - val_AUC: 0.7433
Epoch 22/30
569/569 [==============================] - 0s 101us/sample - loss: 0.5454 - AUC: 0.7987 - val_loss: 0.5943 - val_AUC: 0.7438
Epoch 23/30
569/569 [==============================] - 0s 100us/sample - loss: 0.5398 - AUC: 0.8057 - val_loss: 0.5926 - val_AUC: 0.7492
Epoch 24/30
569/569 [==============================] - 0s 79us/sample - loss: 0.5328 - AUC: 0.8122 - val_loss: 0.5912 - val_AUC: 0.7493
Epoch 25/30
569/569 [==============================] - 0s 86us/sample - loss: 0.5283 - AUC: 0.8147 - val_loss: 0.5902 - val_AUC: 0.7509
Epoch 26/30
569/569 [==============================] - 0s 67us/sample - loss: 0.5246 - AUC: 0.8196 - val_loss: 0.5845 - val_AUC: 0.7552
Epoch 27/30
569/569 [==============================] - 0s 72us/sample - loss: 0.5205 - AUC: 0.8271 - val_loss: 0.5837 - val_AUC: 0.7584
Epoch 28/30
569/569 [==============================] - 0s 74us/sample - loss: 0.5144 - AUC: 0.8302 - val_loss: 0.5848 - val_AUC: 0.7561
Epoch 29/30
569/569 [==============================] - 0s 77us/sample - loss: 0.5099 - AUC: 0.8326 - val_loss: 0.5809 - val_AUC: 0.7583
Epoch 30/30
569/569 [==============================] - 0s 80us/sample - loss: 0.5071 - AUC: 0.8349 - val_loss: 0.5816 - val_AUC: 0.7605

```


### 4. 모델 평가 


먼저 학습과 검증 데이터셋에서 모델의 성능을 평가합니다. 

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
```

```python
plot_metric(history,"loss")
```

![](../data/1-1-Loss曲线.jpg)

```python
plot_metric(history,"AUC")
```

![](../data/1-1-AUC曲线.jpg)


테스트 데이터 셋에서 성능을 살펴 보겠습니다. 

```python
model.evaluate(x = x_test,y = y_test)
```

```
[0.5191367897907448, 0.8122605]
```

```python

```

### 5. 모델의 응용 

```python
#Predict the possiblities
model.predict(x_test[0:10])
#model(tf.constant(x_test[0:10].values,dtype = tf.float32)) #Identical way
```

```
array([[0.26501188],
       [0.40970832],
       [0.44285864],
       [0.78408605],
       [0.47650957],
       [0.43849158],
       [0.27426785],
       [0.5962582 ],
       [0.59476686],
       [0.17882936]], dtype=float32)
```

```python
#Predict the classes
model.predict_classes(x_test[0:10])
```

```
array([[0],
       [0],
       [0],
       [1],
       [0],
       [0],
       [0],
       [1],
       [1],
       [0]], dtype=int32)
```

```python

```

### 6. 모델의 저장 


학습된 모델은 Keras의 방법이나, Tensorflow의 방법을 이용하여 저장할 수 있습니다. Tensorflow의 방법이 여러 플랫폼에 배포할 수 있는 반면에, Keras의 방법은 나중에 모델을 불러올 수만 있습니다. 

모델을 저장하는 방법은 Tensorflow의 방법을 추천합니다. 


**(1) Keras로 모델 저장하기**

```python
# Saving model structure and parameters

model.save('../data/keras_model.h5')  

del model  #Deleting current model

# Identical to the previous one
model = models.load_model('../data/keras_model.h5')
model.evaluate(x_test,y_test)
```

```
[0.5191367897907448, 0.8122605]
```

```python
# Saving the model structure
json_str = model.to_json()

# Retrieving the model structure
model_json = models.model_from_json(json_str)
```

```python
# Saving the weights of the model
model.save_weights('../data/keras_model_weight.h5')

# Retrieving the model structure
model_json = models.model_from_json(json_str)
model_json.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )

# Load the weights
model_json.load_weights('../data/keras_model_weight.h5')
model_json.evaluate(x_test,y_test)
```

```
[0.5191367897907448, 0.8122605]
```


**(2) TensorFlow의 원래 방식대로 모델 저장하기**

```python
# Saving the weights, this way only save the tensors of the weights
model.save_weights('../data/tf_model_weights.ckpt',save_format = "tf")
```

```python
# Saving model structure and parameters to a file, so the model allows cross-platform deployment

model.save('../data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('../data/tf_model_savedmodel')
model_loaded.evaluate(x_test,y_test)
```

```
[0.5191365896656527, 0.8122605]
```


Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```

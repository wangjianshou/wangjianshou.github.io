---
layout: post
title:  "tf2"
date:   2019-11-24
categories: jekyll update
---

[toc]

#### 第一种建立model的方法
```
from tensorflow import keras
inputs = keras.Input(shape=(784, ))
inputs.shape
# TensorShape([None, 784]),也就是说第一个维度是batch size被自动添加

from tensorflow.keras import layers
inputs = keras.Input(shape=(784,), name='img')
x = layers.Dense(64, activation='relu')(inputs) # 调用了__call__方法
x = layers.Dense(64, activation='relu')(x)  # 调用了__call__方法，输入是上一层的输出
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

model.sumary() # 查看图的概况
# keras.utils.plot_model(model, 'my_first_model.png')
keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)
# 图形展示出model的概况

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
```

#### 第二种建立model的方法
```
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Or activation=tf.nn.relu
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.optimizers.RMSprop(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
               )

model.fit(x_train, y_train, epochs=5, batch_size=64,validation_split=0.2)  # batch_size=32时不收敛
test_scores = model.evaluate(x_test, y_test, verbose=0)

# tf.nn.top_k(model.predict(x_train[:10,:]),1) # 预测
model.predict_classes(x_train[:10,:])
```

```
model.save('path_to_my_model.h5')
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model('path_to_my_model.h5')
```


`model.add`
* 参数activation设置层的激活函数。此函数由内置函数的名称指定，或指定为可调用对象。默认情况下不会应用任何激活函数。
* kernel_initializer,bias_initializer 创建层权重的初始化方案，此参数是一个名称或可调用对象，默认为"Glorot uniform"初始化器。
* kernel_regularizer和bias_regularizer：应用层权重的正则化方案，例如L1和L2正则化。默认情况下，系统不会应用正则化函数。



`model.compile`
* 参数optimizer：从tf.train模块向其传递优化器的实例，tf.optimizers.Adam、tf.optimizers.RMSProp 或 tf.optimizers.SGD等。
* loss：优化期间最小化的函数，由名称或者tf.keras.losses传递可调用对象。常见选择包括均方误差('mse')，'categorical_crossentropy'和'binary_crossentropy'。
*  metrics用于监控训练，它们是tf.keras.metrics模块中的字符串或可调用对象。
```
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.optimizers.Adam(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.optimizers.RMSProp(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```

`model.fit`
* epochs：以周期为单位进行训练，一个周期是对整个输入数据进行一次迭代（以较小的批次完成迭代）
* batch_size：当传递Numpy数组时，模型将分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的样本数量。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。  
如果是tf.data.Dataset对象，则可以不指定这个参数，因为Dataset可以设定batch大小。
* validation_data：指定验证数据集，若指定该参数则会在每一个epoch完成后，输出当前在验证集上的效果。
* steps_for_epoch：每一个周期训练的训练的批次数，默认会把全部样本训练一遍。如果是tf.data则默认会迭代耗尽
* validation_split=0.2: 分出最后0.2的样本作为验证集，每个epoches结束，都继续在该验证集上计算误差。仅支持输入数据为numpy array时，其他想dataset，generator，dataset-iterator等都不支持该参数。
* shuffle=True：每个epoch之前shuffle训练数据
* validation_steps：当指定validation_data，且validation_data是dataset或dataset_iterator时，执行validation的batch数。

#### 建立model的第三种方法
```
class myModel(tf.keras.Model):
  def __init__(self, num_class=10):
    super().__init__(name='my_model')
    self.num_class = num_class
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.softmax = tf.keras.layers.Dense(num_class, activation='softmax')
  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    x = self.dense2(x)
    if training:
      x = self.dropout(x, training=training)
    return self.softmax(x)    

model = myModel()
model.compile(optimizer=tf.optimizers.RMSprop(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
               )

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=0)
```



# tf.keras

## tf.keras.layers
```
tf.keras.layers.Conv2D()
# filters: Integer, 卷积核的数量，比如32
kernel_size: 卷积核的shape，如果是整数则每个维度都是这个长度
# strides, padding等类似于tf.nn.conv2d
# input_shape 每个样本的数据的维度，比如三个通道的图片可以是[28,28,3]。但实际上的训练数据要比input_shape多一个维度，这个维度指定了batch size。比如每个batch大小是50，则训练数据的shape是[50,28,28,3]
# 输出的shape是[50,26,26,32]

tf.keras.layer.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

```

##### tf.keras.layers.Dense
```
tf.keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None)
```
* units: 输出神经元个数
* activation：激活函数
* use_bias: Boolean, whether the layer uses a bias vector
* kernel_initializer: Initializer for the `kernel` weights matrix.
* bias_initializer: Initializer for the bias vector.
* kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
* bias_regularizer: Regularizer function applied to the bias vector.
* `Dense` implements the operation: `output = activation(dot(input, kernel) + bias)`；因此如果是输入没有经过Flatten，也就是多个维度，那么输出本层相当于进行了一次`$1\times 1$`的卷积；
* input一行是一个样本，那么kernel的行数就是输入神经元的个数，kernel的列数就是输出神经元的个数。
* Input shape: (batch_size, input_dim)
* Output shape: (batch_size, units)
* `__call__(self, inputs, *args, **kwargs)`方法：会调用`call`方法，返回输出张量
* 如果本层在第一层，可以通过参数`input_dim=512`指定输入的长度.

##### tf.keras.layers.Embedding
```
tf.keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform',input_length=None)
# for example
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
input_array = np.random.randint(1000, size=(32, 10))
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
# 使用与训练的嵌入向量，并设置为不可训练
```
* `input_dim`是单词表的长度，`output_dim`是嵌入向量的长度，`input_length`：仅截取每个样本的前`input_length`个词
* 输入是一个batch的是数据，其中的每一个词是其在单词表中的下标；输出加了一个维度，就是单词用嵌入向量表示了。
* 也可以是某一个单词的下标，返回该单词的嵌入向量
* `Embedding.get_weights()`返回的是该层的参数，`shape=[input_dim, output_dim]`，每一行对应了一个单词的嵌入向量（跳字模型的中心词向量）；


##### tf.keras.layers.GlobalAveragePooling1D
```
layers.GlobalAveragePooling1D(data_format='channels_last')

```
* 默认输入为`(batch, steps, features)`，在`steps`维度上做池化；形象点就是在这一批次的词中，在每一个词向量的同一维度上做池化。

##### tf.keras.layers.Flatten
```
x = np.arange(27).reshape([3,3,3])
layers.Flatten()(x)
#结果是两个维度，第一个维度是batch_size
```

##### tf.keras.layers.SimpleRNN
```
# layers.SimpleRNN(units, return_sequences)
rnn = layers.SimpleRNN(12)
x = np.random.random([10,8,3]).astype('float32')
rnn(x)  # shape为[10,12]

# 解释了第一个样本，第二个时间步输出的计算过程；
rnns = layers.SimpleRNN(12, return_sequences=True)
rnns(x)  # shape为[10,8,12]

a = np.dot(x[0,1,], rnns.get_weights()[0])
b = np.dot(rnns(x)[0,0,], rnns.get_weights()[1])
r = tf.nn.tanh(a + b + rnns.get_weights()[2])
r == rnns(x)[0,1,]
```
* 零时间步的初始状态为全0向量，之后每一步的输出是下一个时间步的状态；
* rnn的输入shape为`[batch_size, timesteps, input_features]`; 其中timesteps的每个词是一个时间步；input_features实际上就是嵌入向量；
* rnn的输出默认shape为`[batch_size, output_features]`，这里的output_features实际上就是参数中的units，也就是输出的神经元个数；这里只输出了最后一个时间步的输出；
* 如果设置为参数`return_sequences=True`，则输出的shape为`[batch_size, timesteps, output_features]`，也就是每个时间步的输出全都输出；如果作为中间层，则应该设置`return_sequencese=True`；
* `w = rnn.get_weights()`；`w[0].shape==(3,12); w[1].shape==[12,12]; w[2].shape==(12,)`；
* `self.build(input_shape=[])`，初始化权重，第一个权重的第一个维度是`input_shape[-1]`，第二个维度是输出的维度units


##### tf.keras.layers.LSTM
与`layers.SimpleRNN`层类似，参考循环神经网络一节。
* dropout 指定该层输入单元的dropout比率；recurrent_dropout指定循环单元的droupout比率。
* 参考
>https://blog.csdn.net/u013006675/article/details/81130452


##### tf.keras.layers.GRU
与`layers.SimpleRNN`层类似，参考循环神经网络一节。

##### tf.keras.layers.Bidirectional
双向RNN利用的RNN的顺序敏感性：它包含两个普通RNN，每个RNN分别沿一个方向对输入序列进行处理（时间正序和时间逆序），然后将它们的表示合并到一起（concat）。通过沿这两个方向处理序列，双向RNN能捕捉到可能被单向RNN忽略的模式。
```
max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
model = keras.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, ecpoches=10, batch_size=128, validation_split=0.2)

```
##### tf.keras.layers.Conv1D
一维卷积神经网络用于文本和序列
```
layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None)
```
* 输入的形状是(batch_size, time, features)，在时间轴上做卷积；

##### tf.keras.layers.MaxPooling1D
```
layers.MaxPooling(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

##### tf.keras.layers.Conv2DTranspose
原理参考[这篇文献](https://arxiv.org/pdf/1603.07285.pdf)，在第20页，4.1节


## tf.keras.layers.Layer

定制了一个Linear层，并建立一个两个层的线性模型，第一层规范化数据，第二层是L2线性回归。
```
# from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
#from sklearn import preprocessing as prep
import matplotlib.pyplot as plt

boston = datasets.load_boston()
#x_boston = prep.StandardScaler().fit_transform(boston.data)

# 为了打乱样本顺序
a = np.arange(506)
np.random.shuffle(a)
x_boston,y_boston = boston.data[a],boston.target[a]

class Linear(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__(**kwargs)
    self.units = units
  def build(self, input_shape):
    self.w = self.add_weight(
               shape=(input_shape[-1], self.units),
               initializer='glorot_normal',
               trainable=True,
               regularizer=tf.keras.regularizers.l1(0.5)
             )
    self.b = self.add_weight(shape=(self.units,),
                initializer = 'glorot_normal',
                trainable=True,
                # regularizer=tf.keras.regularizers.l1(0.1)
                )
    self.built = True  # 也就是说权重只初始化一次
  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

model = keras.Sequential()
model.add(keras.layers.BatchNormalization())
model.add(Linear(1))
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),loss='mse', metrics=['mse'])

model.fit(x_boston[:406,:], y_boston[:406],epochs=1000, batch_size=64,validation_data=(x_boston[406:,:], y_boston[406:]))

test_scores = model.evaluate(x_boston[406:,:], y_boston[406:], verbose=0)

plt.scatter(y_boston[406:], model.predict(x_boston)[406:,:], marker='.')
plt.scatter(y_boston[406:], model.predict(x_boston)[406:,:], marker='.')
plt.show()

```
**关于定制的Linear层**  
* `__init__`：可以执行与输入无关的初始化；
* `build`：按照输入张量的shape初始化权重，也可以进行其他的初始化；
* `call`：进行正向计算。


* 第一次调用`__call__`时会首先调用`build`，建立权重；之后调用`call`进行运算；  
* `call`不会自动调用`build`，因此在手动调用`call`之前必须保证权重张量已经存在了；
* 用`build`而不是`__init__`初始化权重是好处是：可以不必过早的指定输入数据的维度，而是在需要计算的时候指定输入数据，再根据输入数据确定权重的shape，初始化权重。也就是可以直到调用`model.fit`方法进行训练时，才根据输入数据的shape初始化权重。

**关于Layer的help信息如下，help(tf.keras.layers.Layer)：**
* `__init__()`: Save configuration in member variables
* `build()`: Called once from `__call__`, when we know the shapes of inputs and `dtype`. Should have the calls to `add_weight()`, and then call the super's `build()` (which sets `self.built = True`, which is nice in case the user wants to call `build()` manually before the first `__call__`).
 * `call()`: Called in `__call__` after making sure `build()` has been called once. Should actually perform the logic of applying the layer to the input tensors (which should be passed in as the first argument).


## tf.keras.metrics

```
metrics.categorical_crossentropy(y_true, y_pred)  # 计算交叉熵
metrics.categorical_accuracy(y_true, y_pred)  # 识别出y_pred概率最高的类，与y_true相同则但会1，否则返回0
metrics.mse(y_true, y_pred)  # mean((y_true-y_pred)^2)
metrics.mae(y_true, y_pred)  # mean(abs(y_true-y_pred))

m = metrics.Mean()
m.update_state([1,3,5,7]) # 平均值
m.result().numpy()
m.update_state([9])

```

## tf.keras.preprocessing
```
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
# sequences: List of lists, where each element is a sequence.



keras.preprocessing.text.Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000) # 只考虑频率最高的前1000个单词
tokenizer.fit_on_texts(samples)  # 构建单词索引
sequence = tokenizer.fit_on_texts(samples)  # 将每个单词换成其索引组成的列表
texts = tokenizer.sequences_to_texts(sequences)
tokenizer.texts_to_matrix(samples, mode='binary') # 序列转换成one-hot形式，此处的shape为[2,1000]就是该序列出现的单词为1，其他位置为0
tokenizer.word_index # 字典，键是单词，值是在单词表中的下标
tokenizer.word_counts # 有序字典，键是单词，值是单词的频率

```
# keras.callbacks

#### keras.EarlyStopping
```
callbacks.EarlyStopping(monitor='acc', patience=3)
```
监控模型的验证精度，如果验证精度在多于3轮的时间内不再改善，终端训练；

#### callbacks.ModelCheckpoint
```
callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='acc', save_best_only=True)
```
在每轮训练过后保存当前权重，`save_best_only=True`表示如果val_loss没有改善，那么就不需要覆盖模型文件，这就可以是保存的模型始终是训练过程中见到的最好的模型；

#### callbacks.ReduceLROnPlateau
```
callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
```
监控模型的验证损失，如果验证损失在10轮内没有改善，那么就触发这个回调函数，出发是将学习率乘以0.1；

#### callbacks.TensorBoard
```
callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1)
```
`histogram_freq`表示每一轮训练之后记录权重直方图；`embeddings_freq`表示每一轮训练之后记录嵌入结果；













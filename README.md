

Sign Language Recognition with InceptionV3
===


## Requirement
Tensorflow 2
Numpy
PIL
Matplotlib

Detail
---
Data Preprocess:
shuffle
validation set : training set = 1 : 9  



Traning argument & trick :
imagenet的weights
batchsize = 32
epoch = 100
(使用earlystop 大概在30-40就收斂了)
model使用dropout + tanh的組合
optimizer = Adam
learning rate = 1e-4
loss = categorical crossentropy




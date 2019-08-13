#deepak.gupta651@gmail.com
#D@ta2014



##### Replace double spaces with single...

import sys, keras, re, os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

#parameter
path = os.getcwd()+'/'
maxlen = 100 #sequence length
dim = 128

#count
num_train = len(open(path + 'train.txt', 'r').readlines())
num_valid = len(open(path + 'valid.txt', 'r').readlines())
num_test = len(open(path + 'test.txt', 'r').readlines())

#Processing image
def img_processing(data):
    f = open(path + data + '.txt')
    im = []
    i = 0
    z = np.zeros((1,1000))
    z = z.tolist()
    while 1:
        lines = f.readlines()
        if not lines:
            break
        for line in lines:
            line = line.strip()
            if os.path.isfile(path + 'data/' + line + '-full.png') == True:
                img_path = path + 'data/' + line + '-full.png'
                img = image.load_img(img_path, target_size=(299, 299))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x_o = preprocess_input(x)
                y = model_1.predict(x_o)
                y = y.tolist()
                im.append(y)
            else:
                im.append(z)
                print('Picture NO.', i+1, ' could not be found.')
            i += 1
            if data == 'train':
                num_arrow = int(i * 50 / num_train) + 1
                percent = i * 100.0 / (num_train)
            elif data == 'Valid':
                num_arrow = int(i * 50 / num_valid) + 1
                percent = i * 100.0 / (num_valid)
            else:
                num_arrow = int(i * 50 / num_test) + 1
                percent = i * 100.0 / (num_test)
            num_line = 50 - num_arrow
            process_bar = data +': [' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
            sys.stdout.write(process_bar)
            sys.stdout.flush()
    im = np.array(im)
    im = np.squeeze(im)
    np.save(path + data + '_im.npy', im)
    print('\n')
    f.close()

#Image feature
print('Turning images into vectors...')
if os.path.isfile(path + 'valid_im.npy') == False:
    model_1 = InceptionResNetV2(weights='imagenet')  #The TF and Keras versions are related to this code.
    img_processing('train')
    img_processing('valid')
    img_processing('test')

#Read image feature
trainimg_feature = np.load(path + 'train_im.npy')
validimg_feature = np.load(path + 'valid_im.npy')
testimg_feature = np.load(path + 'test_im.npy')

list_code, train_code, valid_code, test_code = [], [], [], []

f = open(path + 'train.txt')
lines = f.readlines()
for line in lines:
    line = line.strip()
    ff = open(path + 'data/' + line + '.html')
    lineff = ff.readlines()
    x = ''
    for ln in lineff:
        x += ln
    x = x.replace('\n', '')
    train_code.append(x)
    ff.close()
f.close()

print()

f = open(path + 'valid.txt')
lines = f.readlines()
for line in lines:
    line = line.strip()
    ff = open(path + 'data/' + line + '.html')
    lineff = ff.readlines()
    x = ''
    for ln in lineff:
        x += ln
    x = x.replace('\n', '')
    valid_code.append(x)
    ff.close()
f.close()

list_code =  train_code + valid_code

dic_code = []
f = open(path+'vocab.txt')
lines = f.readlines()
for i in lines:
    dic_code.append(i.strip())
            
dic_code = len(dic_code)+1

list_code = set(list_code)

 
tokenizer_c = text.Tokenizer(filters='', lower=False, split=' ')
tokenizer_c.fit_on_texts(list_code)
traincode_feature = tokenizer_c.texts_to_sequences(train_code)

validcode_feature = tokenizer_c.texts_to_sequences(valid_code)
traincode_feature = sequence.pad_sequences(traincode_feature, maxlen, padding='post', value=0, truncating='post')
validcode_feature = sequence.pad_sequences(validcode_feature, maxlen, padding='post', value=0, truncating='post')

traincode_hot = keras.utils.to_categorical(traincode_feature, dic_code)  
validcode_hot = keras.utils.to_categorical(validcode_feature, dic_code)
#------------------------------------------------------------------------------------------------------


#image
encoded_image = Input(shape=(1000,))
repeat_image = RepeatVector(maxlen)(encoded_image)
#batch_model = keras.layers.BatchNormalization()(batch_model)
#batch_model1 = Flatten()(batch_model)

batch_model = Dense(maxlen)(repeat_image)
batch_model = Permute((2, 1))(batch_model)
#----------------------------------------------------------------------------

output_model = TimeDistributed(Dense(dic_code, activation='softmax'))(batch_model)
pix2code_model = Model(inputs=[encoded_image], outputs=output_model)
pix2code_model.summary()

#------------------------------------------------------------------------------------------------------

#compile
adam = optimizers.Adam()

#------------------------------------------------------------------------------------------------------
pix2code_model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=[metrics.categorical_accuracy])
#------------------------------------------------------------------------------------------------------


#save
filepath = path + 'MODEL.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


#------------------------------------------------------------------------------------------------------
#train
if os.path.isfile(path + 'MODEL.hdf5') == False:
    print('Training model...')
    history = pix2code_model.fit([trainimg_feature],[traincode_hot], epochs=1000, batch_size=5, validation_data=([validimg_feature],[validcode_hot]), callbacks=callbacks_list, verbose=1)
    
#------------------------------------------------------------------------------------------------------
    
        #Visual training process
    fig = plt.figure()
    fig.set_dpi(300)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    fig.savefig(path + 'MODELA.png')
    
    fig = plt.figure()
    fig.set_dpi(300)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    fig.savefig(path + 'MODELL.png')
    
    # serialize model to JSON
    model_json = pix2code_model.to_json()
    with open(path + 'MODEL.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    pix2code_model.save_weights(path + 'MODEL.h5')
    print("Saved model to disk")
#------------------------------------------------------------------------------------------------------
   
else:

    print('Loading model...')  
    # load json and create model
    json_file = open(path + 'MODEL.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    pix2code_model = model_from_json(loaded_model_json)
    # load weights into new model
    pix2code_model.load_weights(path + 'MODEL.h5', by_name=True) 
    print("Loaded model from disk")
#------------------------------------------------------------------------------------------------------
#evaluation
print('Code generating...')
dic_c = tokenizer_c.word_index
ind_c ={value:key for key, value in dic_c.items()}
#------------------------------------------------------------------------------------------------------

#Generate code
def final_code(data, num):
    if data == 'valid':
        cod = pix2code_model.predict([validimg_feature])
    else:
        cod = pix2code_model.predict([testimg_feature])
    fp = open(path + data + '.predcod', 'w')
    for h in range(num):
        i = h
        if np.argmax(cod[i][0],axis=0) == 0:
            fp.write('abnormality\n')
        else:
            for j in range(maxlen):
                an = np.argmax(cod[i][j],axis=0)
                if j != maxlen-1:
                    anext = np.argmax(cod[i][j+1],axis=0)
                    if an != 0 and anext != 0:  
                        if an != anext:
                            fp.write(ind_c[an] + ' ')
                    elif an != 0 and anext == 0:  
                        fp.write(ind_c[an])
                    elif an == 0 and anext != 0:  
                        fp.write(' ')
                else:
                    if an != 0:
                        fp.write(ind_c[an] + '\n')
                    else:
                        fp.write('\n')
    fp.close()
   
final_code('valid', len(validimg_feature))
final_code('test', len(testimg_feature))

print('Finished.')

#---------------------------------------------------------------------------

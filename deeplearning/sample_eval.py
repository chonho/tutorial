import sys, os
from keras.preprocessing import image
import numpy as np
import sample_cnn

nb_epoch = 100 

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('enter [image file]')
        sys.exit()

    img_file = sys.argv[1]

    h = 64 
    w = 64 
    nb_class = 3 

    ckpt_file = os.path.join(result_dir, 'ckpt-samplecnn-weight.h5')

    #--- construct and load model ---
    model = sample_cnn.smallcnn(h, w, nb_class, ckpt_file)

    # load image and transform it to tensor
    img = image.load_img(img_file, target_size=(h, w))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = x / 255.0

    pred = model.predict(x)[0]
    
    print(np.argmax(pred))

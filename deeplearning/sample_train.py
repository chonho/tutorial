import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import sample_cnn

nb_epoch = 100 

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tval_loss\n")
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\n" % (i, loss[i], val_loss[i]))
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

if __name__ == '__main__':

    h = 64 
    w = 64 
    nb_class = 3 

    ckpt_file = os.path.join(result_dir, 'ckpt-samplecnn-weight.h5')

    #--- construct model ---
    model = sample_cnn.smallcnn(h, w, nb_class)
    #model = sample_cnn.smallcnn(h, w, nb_class, ckpt_file)
    model.summary()

    # generator for training/validation data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(h, w),
        batch_size=16,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        target_size=(h, w),
        batch_size=16,
        class_mode='categorical')

    print(train_generator.class_indices)

    #--- training ---
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=80,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=20
    )

    # Save the results
    model.save_weights(os.path.join(result_dir, 'ckpt-samplecnn-weight.h5'))
    save_history(history, os.path.join(result_dir, 'ckpt-samplecnn-history.txt'))


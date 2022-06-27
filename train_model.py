import os
from urllib import request
import zipfile
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, classification_report
import architectures


### 1. Download data
data_path = "data/"
if not os.path.exists(data_path+"train/") and not os.path.exists(data_path+"test/"):
    print("Downloading training set...")
    response = request.urlretrieve("https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/QHV7PJ", data_path+"train.zip")
    with zipfile.ZipFile(data_path+"/train.zip", 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Downloading test set...")    
    response = request.urlretrieve("https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/Z7JPFT", data_path+"test.zip")
    with zipfile.ZipFile(data_path+"/test.zip", 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print("Done.")


### 2. Create tf.Dataset objects
image_size = (288+512, 288+512)
batch_size = 16
n_pos = len(next(os.walk(data_path+"train/pos"))[2])
n_neg = len(next(os.walk(data_path+"train/neg"))[2])
print("Positive samples: {} ({:d}%), Negative samples: {} ({:d}%)".format(
    n_pos, int(n_pos/(n_pos+n_neg)*100), n_neg, int(n_neg/(n_pos+n_neg)*100)))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path+"train",
    label_mode = "categorical",
    image_size=image_size,
    batch_size=1,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path+"test",
    label_mode = "categorical",
    image_size=image_size,
    batch_size=batch_size,
)

# Balance the positive and negative classes in the training set with oversampling
negative_ds = (
  train_ds
    .unbatch()
    .filter(lambda features, label: tf.argmax(label)==0)
    .repeat())

positive_ds = (
  train_ds
    .unbatch()
    .filter(lambda features, label: tf.argmax(label)==1)
    .repeat())

oversampled_ds = tf.data.experimental.sample_from_datasets([negative_ds, positive_ds], weights=[0.5, 0.5])
oversampled_ds = oversampled_ds.batch(batch_size)

# Prefetch allows later elements to be prepared while the current element is being processed. 
oversampled_ds = oversampled_ds.prefetch(buffer_size=batch_size)
test_ds = test_ds.prefetch(buffer_size=batch_size)


### 3. Create model
model = architectures.custom_Xception_model(input_shape=image_size + (3,), num_classes=2)
model.summary()

METRICS = [
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

model.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=METRICS
)


### 4. Train the model
epochs = 200
oversample_steps_per_epoch = np.ceil(2.0*n_neg/batch_size)
metric_to_monitor = 'val_loss'

callbacks = [
    keras.callbacks.ModelCheckpoint("models/trained_model.h5", monitor=metric_to_monitor, save_best_only=True),
    keras.callbacks.EarlyStopping(monitor=metric_to_monitor, patience=20, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.5, patience=10)
]

training_history = model.fit(
    oversampled_ds,
    epochs=epochs, 
    callbacks=callbacks, 
    validation_data=test_ds, # for simplicity, in this example we use the test as validation
    steps_per_epoch=oversample_steps_per_epoch,
)


### 5. Testing
y_true = []
y_pred = []
for x, y in test_ds:
    y_true.append(y.numpy().argmax(axis=-1))
    y_pred.append(model.predict(x).argmax(axis=-1))
y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
    
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
print("F1 score: {:.2f}, Acc: {:.2f}".format(f1, acc)) 
print(classification_report(y_true, np.round(y_pred)))
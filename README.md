This repository contains an example of how to train the deep learning architecture and how to use the interpretability tools used in the paper [Recognition of polar lows in Sentinel-1 SAR images with deep learning](https://arxiv.org/abs/2203.16401).

----
### Dataset

The *Sentinel-1 maritime mesocyclone dataset* is publicly available and can be downloaded [here](https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/FV5T9U&version=1.0).


----
### Installation (with Anaconda)

Create an Anaconda environment using the [environment.yml](https://github.com/FilippoMB/Recognition-of-polar-lows-in-Sentinel-1-SAR-images-with-deep-learning/blob/main/environment.yml) file.

```
conda env create -f environment.yml
```

The environment was created on Ubuntu 20.04.
For more details of how to create and manage an environment, look [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

----
### Pretrained models

The best performing models can be downloaded [here](https://drive.google.com/drive/folders/1qwFv4qwEfRYj5IEO-vYJwlaqwMFaut_v?usp=sharing).

To use a pretrained model:
```python
import tensorflow as tf
import tensorflow_addons as tfa

model = tf.keras.models.load_model(
    'models/model_1536_F1_095.h5', 
    custom_objects={'AdamW': tfa.optimizers.AdamW})
img = tf.keras.preprocessing.image.load_img(
    "data/test/pos/cffe42_20191012T084028_20191012T084212_mos_rgb.png", 
    target_size=(1536, 1536))
img_array = tf.keras.preprocessing.image.img_to_array(img)

pred = model.predict(tf.expand_dims(img_array, 0))
print("Predicted class: ", pred[0])
```

----
### Model training from scratch
If you want to train the deep learning model from scratch take a look at [train_model.py](https://github.com/FilippoMB/Recognition-of-polar-lows-in-Sentinel-1-SAR-images-with-deep-learning/blob/main/train_model.py), which provides a simple example of how to train the architecture adopted in our paper. The script downloads automatically the dataset in the ```data/``` folder.

----
### Model interpretability

The following notebooks show how to use the interpretability techniques to see what the deep learning model is focusing on:
- [GradCAM.ipynb](https://github.com/FilippoMB/Recognition-of-polar-lows-in-Sentinel-1-SAR-images-with-deep-learning/blob/main/GradCAM.ipynb)
- [Integrated_gradients.ipynb](https://github.com/FilippoMB/Recognition-of-polar-lows-in-Sentinel-1-SAR-images-with-deep-learning/blob/main/Integrated_gradients.ipynb)

----
### Citation
Consider citing our paper if you are using our dataset in your reasearch

    @article{grahn2022recognition,
      title={Recognition of polar lows in Sentinel-1 SAR images with deep learning},
      author={Grahn, Jakob and Bianchi, Filippo Maria},
      journal={arXiv preprint arXiv:2203.16401},
      year={2022}
    }

import os
from tabnanny import verbose

import numpy as np
import cv2
import pandas as pd
from glob import glob

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from metrics import dice_coef, dice_loss
from Train import load_data
from Main import unet

W = 256
H = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, asis = -1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, asis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones([H, 10, 3]) * 255

    cat_images = np.concatenate([image, line, mask, y_pred], axis = 1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    np.random.seed(30)
    tf.random.set_seed(30)

    create_dir("results")

    with CustomObjectScope({"dice_coef": dice_coef, "dice_lose": dice_lose}):
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))

    dataset_path = r"C:\Users\P. Abhinav Sai\PycharmProjects\Image Segmentation\Dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total = len(test_y)):
        name = x.split("/")[-1]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        y_pred = model.predict(x, verbose = 0)
        y_pred = np.squeeze(y_pred, axis =- 1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)

        mask = mask / 255.0
        mask = (mask>0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        f1_value = f1_score(mask, y_pred, labels = [0, 1], average = "binary")
        jac_value = jaccard_score(mask, y_pred, labels = [0, 1], average = "binary")
        recall_value = recall_score(mask, y_pred, labels = [0, 1], average = "binary", )
        precision_value = precision_value(mask, y_pred, labels = [0, 1], average = "binary")

        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])



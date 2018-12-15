import numpy as np
import os
import cv2 as cv
import importlib
import Online_SSL.func as funcs_2

importlib.reload(funcs_2)

path = os.getcwd() + "/data/"

funcs_2.create_user_profile("Dimitri")

funcs_2.create_user_profile("Jean-Guy")

funcs_2.create_user_profile("Blairau")


path = "/home/dimitribouche/Bureau/MVA/S1/GML/TP3/code_material_python"


labelled_dimi = funcs_2.load_profile("Dimitri", faces_path=path + "/data/")
labelled_jg = funcs_2.load_profile("Jean-Guy", faces_path=path + "/data/")

test = funcs_2.load_profile("Blairau", faces_path=path + "/data/")

labelled = np.concatenate((labelled_dimi, labelled_jg))
labels = np.ones(labelled.shape[0], dtype=int)
labels[10:] = -1

test_incre = funcs_2.incremental_k_centers(labelled, labels, max_num_centroids=21)

test_incre.online_ssl_update_centroids(test[0])
test_incre.online_ssl_update_centroids(test[1])
test_incre.online_ssl_update_centroids(test[2])
import numpy as np
import os
import cv2 as cv
import importlib
import Online_SSL.func as funcs_2

importlib.reload(funcs_2)

path = os.getcwd() + "/data/"

funcs_2.create_user_profile("Dimitri")

funcs_2.create_user_profile("Soraya")

path = "/home/dimitribouche/Bureau/MVA/S1/GML/TP3/code_material_python"

funcs_2.online_face_recognition(("Dimitri", "Soraya"))
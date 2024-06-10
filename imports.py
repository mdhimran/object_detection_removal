import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.models.detection as models
import numpy as np
import cv2
import requests
from io import BytesIO
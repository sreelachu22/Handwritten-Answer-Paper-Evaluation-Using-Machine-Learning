from .pdf2image import convert_from_path
import cv2
import glob
import numpy as np
from PIL import Image
import difflib
import importlib
import math
import random
import string

import gluonnlp as nlp
import leven
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mxnet as mx
from skimage import transform as skimage_tf, exposure
from tqdm import tqdm

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.sclite_helper import ScliteHelper
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

importlib.reload(ocr.utils.denoiser_utils)
from ocr.utils.denoiser_utils import SequenceGenerator

importlib.reload(ocr.utils.beam_search)
from ocr.utils.beam_search import ctcBeamSearch


from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

def handrecog():
  #Saving pages in jpeg format
  cnt=0
  pages = convert_from_path('/content/drive/MyDrive/answer sheets pdf/testimage.pdf', 500)
  for i,page in enumerate(pages):
    y="/content/drive/MyDrive/images/"+str(i)+"im.jpeg"
    page.save(y, 'JPEG')
    cnt+=1
  print("No.of pages:",cnt)
  images = []
  files = glob.glob ("/content/drive/MyDrive/images/*.jpeg")
  for myFile in files:
      print(myFile)
      image = Image.open(myFile)
      new_image = image.resize((2479, 3542))
      new_image.save(myFile)
      image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
      images.append (image)
  imag=np.array(images)
  print(' shape:', imag.shape)
  random.seed(123)
  ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
  random.seed(1)
  plt.figure(figsize=(10,10)) # specifying the overall grid size

  for i in range(cnt):
      plt.subplot(2,2,i+1)    # the number of images in the grid is 5*5 (25)
      plt.imshow(imag[i], cmap='Greys_r')

  paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
  paragraph_segmentation_net.cnn.load_parameters("models/paragraph_segmentation2.params", ctx=ctx)

  paragraph_segmentation_net.hybridize()

  from matplotlib.patches import Rectangle
  form_size = (1120, 800)

  predicted_bbs = []
  plt.figure(figsize=(10,10)) # specifying the overall grid size
  #fig, axs = plt.subplots(int(len(imag)/2), 2, figsize=(15, 9 * len(imag)/2))
  for i, image in enumerate(imag):
    #s_y, s_x = int(i/2), int(i%2)
    resized_image = paragraph_segmentation_transform(image, form_size)
    bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))
    bb_predicted = bb_predicted[0].asnumpy()
    bb_predicted = expand_bounding_box(bb_predicted, expand_bb_scale_x=0.03, expand_bb_scale_y=0.03)
    predicted_bbs.append(bb_predicted)
    
    #axs[s_y, s_x].imshow(image, cmap='Greys_r')
    #axs[s_y, s_x].set_title("{}".format(i))

    (x, y, w, h) = bb_predicted
    image_h, image_w = image.shape[-2:]
    (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
    rect = patches.Rectangle((x, y), w, h, fill=False, color="r", ls="--")
    #axs[s_y, s_x].add_patch(rect)
  

    plt.subplot(2,2,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(imag[i], cmap='Greys_r')
    plt.gca().add_patch(rect)

    segmented_paragraph_size = (700, 700)
  #fig, axs = plt.subplots(int(len(images)/2), 2, figsize=(15, 9 * len(images)/2))
  plt.figure(figsize=(10,10))
  paragraph_segmented_images = []

  for i, image in enumerate(images):
      #s_y, s_x = int(i/2), int(i%2)

      bb = predicted_bbs[i]
      image = crop_handwriting_page(image, bb, image_size=segmented_paragraph_size)
      paragraph_segmented_images.append(image)
      plt.subplot(2,2,i+1)    # the number of images in the grid is 5*5 (25)
      plt.imshow(image, cmap='Greys_r')
      #axs[s_y, s_x].imshow(image, cmap='Greys_r')
      #axs[s_y, s_x].axis('off')

  word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
  word_segmentation_net.load_parameters("models/word_segmentation2.params")
  word_segmentation_net.hybridize()

  min_c = 0.1
  overlap_thres = 0.1
  topk = 600
  plt.figure(figsize=(10,10))
  #fig, axs = plt.subplots(int(len(paragraph_segmented_images)/2), 2, figsize=(15, 5 * int(len(paragraph_segmented_images)/2)))
  predicted_words_bbs_array = []

  for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):
      #s_y, s_x = int(i/2), int(i%2)
      predicted_bb = predict_bounding_boxes(word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)
      predicted_words_bbs_array.append(predicted_bb)
      #axs[s_y, s_x].imshow(paragraph_segmented_image, cmap='Greys_r')
      plt.subplot(2,2,i+1)    # the number of images in the grid is 5*5 (25)
      plt.imshow(paragraph_segmented_image, cmap='Greys_r')
      for j in range(predicted_bb.shape[0]):     
          (x, y, w, h) = predicted_bb[j]
          image_h, image_w = paragraph_segmented_image.shape[-2:]
          (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
          rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
          plt.gca().add_patch(rect)
          #axs[s_y, s_x].add_patch(rect)
          #axs[s_y, s_x].axis('off')

  line_images_array = []
  #fig, axs = plt.subplots(int(len(paragraph_segmented_images)/2), 2, figsize=(15, 9 * int(len(paragraph_segmented_images)/2)))
  plt.figure(figsize=(10,10))
  for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):
      #s_y, s_x = int(i/2), int(i%2)
      #axs[s_y, s_x].imshow(paragraph_segmented_image, cmap='Greys_r')
      #axs[s_y, s_x].axis('off')
      #axs[s_y, s_x].set_title("{}".format(i))
      plt.subplot(2,2,i+1)    # the number of images in the grid is 5*5 (25)
      plt.imshow(paragraph_segmented_image, cmap='Greys_r')
      predicted_bbs = predicted_words_bbs_array[i]
      line_bbs = sort_bbs_line_by_line(predicted_bbs, y_overlap=0.4)
      line_images = crop_line_images(paragraph_segmented_image, line_bbs)
      line_images_array.append(line_images)
    
      for line_bb in line_bbs:
          (x, y, w, h) = line_bb
          image_h, image_w = paragraph_segmented_image.shape[-2:]
          (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

          rect = patches.Rectangle((x, y), w, h, fill=False, color="r")
          plt.gca().add_patch(rect)
  return 0
  
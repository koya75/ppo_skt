import os
from tqdm import tqdm
import cv2
import numpy as np
import csv
import argparse

import warnings
warnings.simplefilter('ignore')

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow AQT')
parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'encoder', 'decoder', 'sketch_decoder', 'sketch_encoder', 'sketch_action_decoder'], metavar='CUDA', help='Cuda Device')
parser.add_argument('--load-dir', type=str, default='visuals/breakout_rainbow_aqt/epi1/', help='Load data')

# Setup
args = parser.parse_args()

# Print options
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))



print("Make movie: {}".format(args.mode))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
if args.mode == "raw":
  movie_path = os.path.join(args.load_dir, "raw_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 2.0, (128, 128))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "raw_img/", name)) for name in os.listdir(args.load_dir + "raw_img/"))
  for idx in tqdm(range(file_num)):
    raw_img = cv2.imread(args.load_dir + "raw_img/raw_{0:06d}.png".format(idx))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    video.write(raw_img)
  video.release()

elif args.mode == "encoder":
  movie_path = os.path.join(args.load_dir, "encoder_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 2.0, (200, 200))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "image_encoder/", name)) for name in os.listdir(args.load_dir + "image_encoder/"))
  for idx in tqdm(range(file_num)):
    en_att = cv2.imread(args.load_dir + "image_encoder/en_{0:06d}.png".format(idx))
    video.write(en_att)
  video.release()

elif args.mode == "decoder":
  movie_path = os.path.join(args.load_dir, "decoder_movie.mp4")
  video = cv2.VideoWriter(movie_path, fourcc, 2.0, (200, 200))

  file_num = sum(os.path.isfile(os.path.join(args.load_dir + "decoder_act/", name)) for name in os.listdir(args.load_dir + "decoder_act/"))
  for idx in tqdm(range(file_num)):
    en_att = cv2.imread(args.load_dir + "decoder_act/de_{0:06d}.png".format(idx))
    video.write(en_att)
  video.release()
import argparse
from PIL import Image
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm.auto import tqdm

EPS = 1e-7

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def save_image(out_fold, img, fname='image', ext='.png'):
    img = img.detach().cpu().numpy().transpose(1,2,0)
    im_out = np.uint8(img*255.)
    cv2.imwrite(out_fold+fname+ext, im_out[:,:,::-1])

class Demo():
    def __init__(self, args):
        ## configs
        self.device = 'cuda:'+args.gpu if args.gpu else 'cpu'
        self.image_size = args.image_size

        from facenet_pytorch import MTCNN
        self.face_detector = MTCNN(select_largest=True, device=self.device)

    def detect_face(self, im):
        # print("Detecting face using MTCNN face detector")
        try:
            bboxes, prob = self.face_detector.detect(im)
            w0, h0, w1, h1 = bboxes[0]
        except:
            print("Could not detect faces in the image")
            return im

        hc, wc = (h0+h1)/2, (w0+w1)/2
        crop = int(((h1-h0) + (w1-w0)) /2/2 *1.1)
        im = np.pad(im, ((crop,crop),(crop,crop),(0,0)), mode='edge')  # allow cropping outside by replicating borders
        h0 = int(hc-crop+crop + crop*0.15)
        w0 = int(wc-crop+crop)
        return im[h0:h0+crop*2, w0:w0+crop*2]

    def run(self, pil_im, save_path):
        input = np.uint8(pil_im)

        ## face detection
        im = self.detect_face(input)

        h, w, _ = im.shape
        if h < 50 or w < 50:
            im = input
        im = torch.FloatTensor(im /255.).permute(2,0,1).unsqueeze(0)
        # resize to 128 first if too large, to avoid bilinear downsampling artifacts
        if h > self.image_size * 4 and w > self.image_size * 4:
            im = nn.functional.interpolate(im, (self.image_size * 2, self.image_size * 2), mode='bilinear',
                                           align_corners=False)
        im = nn.functional.interpolate(im, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
        save_image(save_path, im[0], '_recrop.png')      

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recrop configurations.')
    parser.add_argument('--input', default='./demo/images/human_face', type=str, help='Path to the directory containing input images')
    parser.add_argument('--gpu', default=None, type=str, help='Enable GPU')
    parser.add_argument('--image_size', default=128, type=int, help='Output image size')
    args = parser.parse_args()

    input_dir = args.input
    model = Demo(args)
    im_list = []
    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if is_image_file(name) and name[-10:] != 'recrop.png':
                im_list.append(os.path.join(root, name))
    # print(im_list)

    for im_path in tqdm(im_list):
        # print(f"Processing {im_path}")
        save_dir = os.path.join(os.path.dirname(im_path), os.path.splitext(os.path.basename(im_path))[0])
        if os.path.exists(save_dir+'_recrop.png'):
            continue

        pil_im = Image.open(im_path).convert('RGB')
        result_code = model.run(pil_im)
        model.save_results(save_dir)

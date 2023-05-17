import streamlit as st
import numpy as np
import pandas as pd

import torchvision
from torchvision import datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau # for lr decay

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

device = torch.device('cpu')

class DenseNet121(nn.Module):
	"""Model modified.
	The architecture of our model is the same as standard DenseNet121
	except the classifier layer which has an additional sigmoid function.
	"""
	def __init__(self, out_size):
		super(DenseNet121, self).__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=True)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
		    nn.Linear(num_ftrs, out_size),
		    nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x


def tensor2nparr(tensor):
	img = tensor.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img = std * img + mean
	img = np.clip(img, 0, 1) # clip 0~225 into 0~1 
	return img


def process_image(image_path):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	'''
	img = Image.open(image_path).convert('RGB')
	transform = transforms.Compose([ transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
	img_tensor = transform(img).float()
	return img_tensor

@st.experimental_memo #avoid reloading data again and again
def load_model():
	model = DenseNet121(2)
	model_path = 'Kaggle_CheXNet_densenet121_5epochs_cuda.pth'
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def chexnet():
	model = load_model()
	
	st.title('ChexNet Pneumonia Classifier')
	with st.sidebar:
		st.write('## Upload a chest x-ray image:')
		uploaded_file = st.file_uploader('', type = (["jpg", "jpeg"]))

	if uploaded_file is not None:
		img = Image.open(uploaded_file)
		#st.image(img, caption = 'Uploaded image')
		st.sidebar.image(Image.open(uploaded_file))#, caption = 'Uploaded image')

		tmp = process_image(uploaded_file)
		img = torch.zeros(1, 3, 224, 224) 
		img[0] = tmp
		img = img.to(device)

		
		with torch.no_grad():
			output = model.forward(img)

		top_p, top_class = output.topk(1)
		idx = top_class.data.item()
		class_name = ['Non-Pneumonia', 'Pneumonia']

		fig, ax = plt.subplots(figsize=(2, 2))
		plt.imshow(tensor2nparr(tmp))
		plt.title(class_name[idx], fontsize=10)
		plt.axis('off')

		st.pyplot(fig)





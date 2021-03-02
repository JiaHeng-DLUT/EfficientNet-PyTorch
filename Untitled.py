#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
#from efficientnet_pytorch import EfficientNet


# In[20]:


#model = EfficientNet.from_pretrained('efficientnet-b7')
# print(model)


# In[21]:


from efficientnet_pytorch.utils import get_model_params
blocks_args, global_params = get_model_params('efficientnet-b7', None)
print(blocks_args)
#[BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=2, kernel_size=3, stride=[2], expand_ratio=6, input_filters=16, output_filters=24, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=24, output_filters=40, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=40, output_filters=80, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=80, output_filters=112, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=112, output_filters=192, se_ratio=0.25, id_skip=True), 
# BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=192, output_filters=320, se_ratio=0.25, id_skip=True)]
print(global_params)


# In[23]:


from efficientnet_pytorch.model import EfficientNet
net = EfficientNet(blocks_args, global_params)
img = torch.ones((1, 3, 224, 224))
print(img.shape)
out = net(img)
print(out.shape)


# In[16]:


class SimpleNet:
    def __init__(self):
        pass
    
    def forward(self, x):
        pass


# In[ ]:





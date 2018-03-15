# Ring Loss
## Getting Started
Install PyTorch and Python.
Download ringloss.py to your working directory. 

## Usage
Initialize a RingLoss module
'''
ringloss_block = RingLoss(type='ratio', loss_weight=1.0)
'''
During forward
'''
ringloss = ringloss_block(feature) # your feature should be (batch_size x feat_size)
'''
During backward, be sure to use ringloss as an augmentation of your classification loss. e.g.
'''
total_loss = softmax_loss + ringloss
total_loss.backward()
'''

## Training
During training, a pretrained model is suggested, since Ring loss may be unstable in the beginning. 

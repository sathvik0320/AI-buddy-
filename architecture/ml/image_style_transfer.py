import numpy as np
import torch
import torchvision as torchv
import matplotlib.pyplot as plt
from PIL import Image


device ="cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


im_size = 512 if torch.cuda.is_available() else 128

new_var = torchv.transforms.Compose([torchv.transforms.ToTensor(),
                                  torchv.transforms.Resize(im_size),
                                  torchv.transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                  ])
transformations = new_var



plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(Image.open("style_transfer/picasso.jpg"))
plt.subplot(1,2,2)
plt.imshow(Image.open("style_transfer/dancing.jpg"))


content_image = (transformations(Image.open("style_transfer/dancing.jpg")).unsqueeze(0))
style_image =(transformations(Image.open("style_transfer/picasso.jpg")).unsqueeze(0))

if content_image.size() != style_image.size():

         resize = torchv.transforms.Resize(im_size)
         content_image=resize(content_image)
         style_image = resize(style_image)

content_image=content_image.to(device)
style_image=style_image.to(device)




class Content_Loss(torch.nn.Module):

  def __init__(self,content_image_features):
    """

    Args:
      target:
    """
    super(Content_Loss,self).__init__()
    self.target = content_image_features.detach()
    self.loss = torch.nn.MSELoss()

  def forward(self,input_image_content):


      return self.loss(input_image_content,self.target)


def Gram_Matrix(input):

  """

  Args:
    input:
  """
  a, b, c, d = input.size()
  features = input.view(a * b, c * d)
  G = torch.mm(features, features.t())  # compute the gram product

  return G.div(a * b * c * d)



class Style_Loss(torch.nn.Module):
  """

  Attributes:
    target:
    loss:
  """

  def __init__(self,style_image_features):
    super(Style_Loss,self).__init__()
    self.target = Gram_Matrix(style_image_features).detach()
    self.loss = torch.nn.MSELoss()

  def forward(self,input_image_style):

     gram_M = Gram_Matrix(input_image_style)

     return self.loss(gram_M,self.target)



#importing the model
model = torchv.models.vgg19(weights=torchv.models.VGG19_Weights.DEFAULT).features.eval()


for parameters in model.parameters():
  parameters.requires_grad_(False)


target_layer = ["conv_4"]


def create_model(model):
  """

  Args:
    model:
    layer:
  """
  seq_model = torch.nn.Sequential()
  i=0
  for layer in model.children():
      if isinstance(layer,torch.nn.Conv2d):
        i=i+1
        name = "conv_{}".format(i)

      elif isinstance(layer,torch.nn.ReLU):
        name = "relu_{}".format(i)
        layer = torch.nn.ReLU(inplace =False)

      elif isinstance(layer,torch.nn.MaxPool2d):
        name = "pool_{}".format(i)

      elif  isinstance(layer,torch.nn.BatchNorm2d):
        name = "bn_{}".format(i)

      else :
        raise RuntimeError("un recognized layer {}".format(layer.__class__.__name__))

      seq_model.add_module(name,layer)



  return seq_model




seq_model = create_model(model)



i=0
for name,layer in seq_model.named_children():
  print(i)
  print(name)
  if name == target_layer[0]:
    break
  else :
    i+=1


print(i)
seq_model = seq_model[:(i+1)]



input_image = torch.randn(content_image.data.size())

image_plot = input_image.squeeze(0).permute(1,2,0).numpy()
#input_image_plot = input_image.clone()
image_plot = (image_plot - image_plot.min()) / (image_plot.max() - image_plot.min())  # Normalization

plt.figure()
#plt.imshow(input_image_plot.squeeze(0).permute(1,2,0).detach().numpy())
plt.imshow(image_plot)





def get_features(seq_model,image_for_features):
  """

  Args:
    seq_model:
    image_for_features:

  Returns:

  """


  seq_model.requires_grad_(False)

  return seq_model(image_for_features)





content_features = get_features(seq_model=seq_model,image_for_features=content_image)
style_features = get_features(seq_model=seq_model,image_for_features=style_image)




content_loss_model = Content_Loss(content_features)
style_loss_model = Style_Loss(style_features)


input_image.requires_grad_(True)

print([input_image])
optimizer = torch.optim.LBFGS([input_image])



epochs = int(input("give number of epochs"))
training_losses = []
num_epochs =[]
for epoch in range(epochs):

  print(epoch)

  def closure():
    """

    Returns:

    """

    with torch.no_grad():
       input_image.clamp_(0, 1)

    alpha = 3.5555 #both are hyper parameters
    beta = 0.001

    optimizer.zero_grad()

    input_features = get_features(seq_model=seq_model,image_for_features=input_image)


    content_loss = content_loss_model(input_features)
    style_loss = style_loss_model(input_features)


    loss =  alpha*style_loss+beta*content_loss

    training_loss = loss.item()

    loss.backward()

    return loss

  optimizer.step(closure)


  training_losses.append(closure().item())
  num_epochs.append(epoch)


with torch.no_grad():
   input_image.clamp_(0, 1)



plt.figure(figsize=(6,6))
plt.plot(num_epochs,training_losses,"blue",label="training_loss")
plt.title("image_style_transfer")
plt.show()



plt.figure(figsize=(5,5))
print(input_image)
image_plot = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalization

plt.imshow((input_image).squeeze(0).permute(1,2,0).detach().numpy())


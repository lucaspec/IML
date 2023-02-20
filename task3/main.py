import numpy as np
import pandas as pd
import os
import torch 
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision import models
from torchvision.datasets import ImageFolder


""" X_data = np.genfromtxt('test_triplets.txt', delimiter=' ')

# save image as array method
def save_to_array(index):
    directory = 'food/' + str(index) + '.jpg'
    image = Image.open(directory)
    array = np.asarray(image)
    return array

print(save_to_array('00002').shape)
 """



# preprocessingg and saving
os.makedirs('newfood', exist_ok = True)
transformationForCNNInput = transforms.Compose([transforms.Resize((224,224))])
i = 0 

for imageName in os.listdir('food'):
    I = Image.open(os.path.join('food', imageName))
    newI = transformationForCNNInput(I)

    # copy the rotation information metadata from original image and save
    if "exif" in I.info:
        exif = I.info['exif']
        newI.save(os.path.join('newfood', imageName), exif=exif)
    else:
        newI.save(os.path.join('newfood', imageName))

    print(i)
    i += 1


# pretrained model
class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()

        # normalising
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        
        return cnnModel, layer
        

# generate vectors for all the images in the set
img2vec = Img2VecResnet18() 

allVectors = {}
print("Converting images to feature vectors:")
for image in tqdm(os.listdir("newfood")):
    I = Image.open(os.path.join("newfood", image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close() 


# define a function that calculates the cosine similarity entries in the similarity matrix
def getSimilarityMatrix(vectors):
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    
    return matrix
        
similarityMatrix = getSimilarityMatrix(allVectors)
similarityMatrix_np = similarityMatrix.to_numpy()
print(similarityMatrix)

# define a function that outputs which image is more similar
def compare(img1, img2, img3):
    if similarityMatrix_np[img1, img2] > similarityMatrix_np[img1, img3]:
        return 1
    else:
        return 0

X_data = np.genfromtxt('train_triplets.txt', delimiter=' ')

X_data = X_data.astype(int)

sum = 0
for i in range(X_data.shape[0]):
   sum += compare(X_data[i,0], X_data[i,1], X_data[i,2])

print(X_data.shape[0])
print(sum)



X_test = np.genfromtxt('test_triplets.txt', delimiter=' ')
X_test = X_test.astype(int)

# write submission file
file = open("submission.txt","a")
print(X_test.shape[0])
j=0
for i in range(X_test.shape[0]):
   res = compare(X_test[i,0], X_test[i,1], X_test[i,2])
   file.write(str(res)+'\n')
   j+=1
   print(j)
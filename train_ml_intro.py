from wildlife_datasets.datasets import SeaTurtleID2022 
import torchvision.transforms as T
import timm
import torch
import numpy as np
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import os
from huggingface_hub import constants, login
#Path to dataset, pre-downloaded dataset, see dl_dataset.py
root = "data/SeaTurtleID2022"

#HF method:
#check env vars
print("HF_HOME =", os.environ.get("HF_HOME"))
print("HUGGINGFACE_HUB_TOKEN =", os.environ.get("HUGGINGFACE_HUB_TOKEN"))
print("HUGGINGFACE_HUB_CACHE =", os.environ.get("HUGGINGFACE_HUB_CACHE"))

print("Resolved HF_HOME =", constants.HF_HOME)
print("Resolved CACHE =", constants.HUGGINGFACE_HUB_CACHE)

#login to huggingface account
login()

#https://wildlifedatasets.github.io/wildlife-tools/inference/#extract-features
print("load the MegaDescriptor-L-384") # (or any other) model.
model_name = "hf-hub:BVRA/MegaDescriptor-L-384"
model = timm.create_model(model_name, num_classes=0, pretrained=True)
torch.save(model.state_dict(), "MegaDescriptor_clean_timm.pth")

#https://wildlifedatasets.github.io/wildlife-tools/inference/#create-dataset
print("transform")
#build transform 
transform = T.Compose([
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

print("getting dataset")

#SeaTurtleID2022.get_data(root) #if not pre-downloaded dataset
dataset = SeaTurtleID2022(
    root,
    transform=transform,
    load_label=True,
    factorize_label=True,
)

'''
print("show dataset[0]")
dataset[0]

print("plot grid")
dataset.plot_grid();
'''

#Split the dataset into the database and query sets. 
#The following split is performed so that both sets contains two individuals, each with ten images.
idx_train = list(range(10)) + list(range(190,200))
idx_test = list(range(10,20)) + list(range(200,210))
dataset_database = dataset.get_subset(idx_train)
dataset_query = dataset.get_subset(idx_test)

#The class SeaTurtleID2022 may then be used for example for feature extraction
print("start feature extraction")
extractor = DeepFeatures(model, batch_size=4, device='cuda')

#query and database are of shape 20xn, 
# where n depends on the model and is the size of the feature (embedding) vector.
query, database = extractor(dataset_query), extractor(dataset_database)

#Calculate cosine similarity between query and database deep features.
#https://wildlifedatasets.github.io/wildlife-tools/inference/#calculate-similarity
similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)

#https://wildlifedatasets.github.io/wildlife-tools/inference/#evaluate
#Use the cosine similarity in nearest neigbour classifier and get predictions.
print("use Knn classifier")
classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
predictions = classifier(similarity)

print("get accuracy")
accuracy = np.mean(dataset_query.labels_string == predictions)
print("h")
# nndst
## Basic Usage
### Prerequisites
The code has the following dependencies:

- python 3.7
- pytorch
- torchvision
- Pillow (PIL)
- scipy 
- networkx

### to generate networkx directed network graph for both structured (EB) and unstructured (FreeTickets) do:
````
bash run.sh
````
#### You will be ask to answer the following prompts:
* "Cuda: " -> determines which gpu to run on
* "Dataset: " -> select dataset(s) to use [cifar10, cifar1000, imagenet]
* "Models: " -> select models to use (consult supported models in common_models/models.py)
* "batchsize: " -> select batch-size to train
* "Epochs: " -> select total epochs
* "seed: " -> set seed for random generator (important as it will also be use to genreate shared random init weights values for both structured and unstructured)
* "sparsity: " -> set sparsity level for Dynamic sparsity training (0.1-1.0)
* "ouput graph dir: " -> output dir to save networkx's gml graphs

#### Sample prompts
* "Cuda: " 6
* "Dataset: " cifar10, cifar1000
* "Models: " resnet18 resnet34 resnet50 vgg16
* "batchsize: " 512
* "Epochs: " 100
* "seed: " 69
* "sparsity: " 0.3
* "ouput graph dir: " graphs/


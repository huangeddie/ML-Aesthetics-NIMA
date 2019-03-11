from models.densenet import DenseNet
from models.vector_quantization import VQModel
from models.demo import Linear
import argparse

available_models = list(map(lambda model: model.__name__, [DenseNet, VQModel, Linear]))

# Parse arguments
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--epochs', '-e', type=int, help='Number of epochs', default=1)
parser.add_argument('--version_name', '-v', type=str, default='default', help='Name of version')
parser.add_argument('--model', '-m', type=str, help='Choose model: {}'.format(available_models))
parser.add_argument('--skip_train', '-st', action='store_true', default=False, help='Skip the training')
args = parser.parse_args()

epochs = args.epochs
vers_name = args.version_name
model_name = args.model
skip_train = args.skip_train

# Do stuff
Model = eval(model_name)
print('Training {}'.format(Model.__name__))

model = Model()
print('Loading {}'.format(vers_name))
model.load(vers_name)
if not skip_train:
    model.train(epochs)
else:
    print('Skipped training...')
model.create_version(vers_name)

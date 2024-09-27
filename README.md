# Neural Field Image Approximation
This project demonstrates the use of Neural Fields to approximate images using a simple fully connected neural network (MLP). The neural network learns to reproduce image structures through a continuous representation, with optional positional encoding to enhance performance.

## Features
- Flexible architecture: Set the number of inputs, hidden layers, and neurons per layer.
- Optional positional encoding to improve the network's ability to learn fine details in images.
- Training and evaluation of the model to overfit an image, allowing visual comparison between the original and generated image.
- Supports both CPU and GPU training.

## Installation 
1. Clone the repository:
```bash
git clone https://github.com/Mikecraft1224/NeuralFields.git
cd NeuralFields
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Ensure you have an image for training (e.g., `ExampleImages/lena_color.png`). You can download any image  you like.
4. (Optional) If you want to speedup training, install CUDA from https://developer.nvidia.com/cuda-downloads. The installed PyTorch version is already compatible with CUDA 12.4.

## Usage
### Training
To train the model on an image:
```bash
python main.py -t -n <model_name> -i <image_path> -e <num_epochs> -s <image_steps> -r <resolution> -d <dimension>
```
- -t: Indicates training mode.
- -n: Name of the model (a folder with this name will be created in the directory).
- -i: Path to the image.
- -e: Number of epochs.
- -s: Number of steps to save the image during training. This will be used to generate a gif of the training process.
- -r: Resolution of the image (e.g., 128 for 128x128).
- -d: Dimension of the positional encoding (0 for no positional encoding).

Example:
```bash
python main.py -t -n lena -i ExampleImages/lena_color.png -e 1000 -s 100 -r 128 -d 0
```

By default this will use 8 hidden layers with 256 neurons per layer. You can change these values by modifying the flags `main.py` file.

### Generating an Image
To generate an image using a trained model:
```bash
python main.py -g -n <model_name> -r <resolution> -d <dimension>
```

Example:
```bash
python main.py -g -n lena -r 128 -d 0
```

### Training multiple models
To train multiple models with different configurations, you can write a configuration file with the desired parameters and run the following command:
```bash
python main.py -q -c <config_file>
```

Example:
```bash
python main.py -q -c config.json
```

Look at the example in the ExampleConfigs folder to see how to write a configuration file or run `python main.py -h` to see an example config line.

## Positional Encoding
Positional encoding can be added to the model by specifying an encoding dimension in the code. For each dimension d, a vector (x,y) is expanded to include additional sine and cosine transformations.

## Project Structure
- `main.py`: Main script to train and generate images.
- `toGif.py`: Script to generate a gif from the images saved during training.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Pillow
- Matplotlib

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
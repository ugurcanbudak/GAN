# GAN Project

This project implements a Generative Adversarial Network (GAN) for generating synthetic data. The GAN consists of a generator network and a discriminator network that compete against each other in a two-player minimax game.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ugurcanbudak/GAN.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the GAN:

    ```bash
    python train.py
    ```

2. Generate synthetic data:

    ```bash
    python generate.py
    ```

## Project Structure

- `train.py`: Script for training the GAN.
- `generate.py`: Script for generating synthetic data using the trained GAN.
- `generator.py`: Contains the implementation of the generator network.
- `discriminator.py`: Contains the implementation of the discriminator network.
- `utils.py`: Utility functions for data preprocessing and visualization.
- `data/`: Directory for storing the training data.
- `images/`: Directory for storing the generated synthetic data.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
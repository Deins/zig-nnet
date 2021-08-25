# WIP: Basic feed-forward neural network with back propagation in zig language

This is zig experiment of writing basic feed forward neural network with back-propagation. It is built and tested for digit recognition on MNIST dataset.

**Note:** This is my first time writing or using neural network, therefore there might be some math errors etc. If you see any, please let me know. 🙂

## Build

```sh
zig build -Drelease-fast
```

Output program should be built at default location `./zig-out/bin/nn` or `*.exe` etc. depending on platform.

## Usage

(Optional) It is recommended to preprocess input for faster further loading. (has to be done once or after dataset or file format is modified).

```sh
./zig-out/bin/nn preprocess
```

Train new network:

```sh
 .\zig-out\bin\nn.exe --save data/n.net --learn-rate 0.5 --batch-size 32 --epoches 2 --workers 4 train
```

You can use `--load data/n.net` afterwards, to load existing net instead of generating new and train it further or with different settings.

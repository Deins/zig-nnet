# WIP: Basic feed-forward neural network with back propagation in zig language

This is zig experiment of writing basic feed forward neural network with back-propagation. It is built and tested for digit recognition on MNIST dataset.

**Note:** This is my first time writing or using neural network, therefore there might be some math errors etc. If you see any, please let me know. 🙂


## Setup

```sh
$ git clone git@github.com:Deins/zig-nnet.git
$ git submodule init
$ git submodule update
```

## Build

```sh
zig build -Drelease-fast
```

Output program should be built at default location `./zig-out/bin/nn` or `*.exe` etc. depending on platform.

## Usage

Download MNIST dataset from [here](https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-digits/#ProblemStatement) and extract it at `./data/digits/`.

(Optional) It is recommended to preprocess input for faster further loading. (has to be done once or after dataset or file format is modified).

```sh
./zig-out/bin/nn preprocess
```

Train new network and save output net:

```sh
 ./zig-out/bin/nn.exe --save data/n.net --learn-rate 0.01 --batch-size 64 --epoches 5 --workers 4 train
```

You can use `--load data/n.net` afterwards, to load existing net instead of generating new and train it further or with different settings.

Test trained model on test data set, and save classified results inspection:
```sh
./zig-out/bin/nn.exe --load data/n.net --img-dir-out ./test-out test
```

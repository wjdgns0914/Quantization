python main.py --model MNIST_MLP_BNN_256_10
python main.py --model MNIST_MLP_BNN_256_10 --Variation True
python main.py --model MNIST_MLP_BNN_256_10 --Drift2 True --Variation True
python main.py --model MNIST_MLP_BNN_512_10
python main.py --model MNIST_MLP_BNN_512_10 --Variation True
python main.py --model MNIST_MLP_BNN_512_10 --Drift2 True --Variation True
python main.py --model MNIST_MLP_BNN_256_256_10
python main.py --model MNIST_MLP_BNN_256_256_10 --Variation True
python main.py --model MNIST_MLP_BNN_256_256_10 --Drift2 True --Variation True

python main.py --model CIFAR10_00_VGG4_BC_A
python main.py --model CIFAR10_00_VGG4_BC_A --Drift1 True --Variation True
python main.py --model CIFAR10_00_VGG4_BC_A --Drift2 True --Variation True
python main.py --model CIFAR10_00_VGG4_BC_A --Drift1 True --Drift2 True --Variation True
python main.py --model CIFAR10_01_VGG4_BAN_A

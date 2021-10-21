# Download the cifar10 dataset
mkdir -p data/cifar10
wget -P 'data/cifar10' https://github.com/lin-tan/fairness-variance/releases/download/dataset/cifar-10-python.tar.gz
cd data/cifar10
tar -xzf cifar-10-python.tar.gz
mv cifar-10-batches-py/* .
rm -rf cifar-10-batches-py
rm cifar-10-python.tar.gz

# Download the cinic dataset
cd ../..
mkdir -p data/cinic
wget -P 'data/cinic' --no-check-certificate https://github.com/lin-tan/fairness-variance/releases/download/dataset/CINIC-10.tar.gz
cd data/cinic
tar -xzf CINIC-10.tar.gz
rm CINIC-10.tar.gz

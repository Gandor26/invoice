# invoice
Invoice processing in Appfolio

Using CNN (AlexNet based) and BoW of vendor names and addresses

## Requirements
Run experiments under Python 3.6.5 with following libs:

* pytorch 0.4.1 with CUDA 9.0 support (optional)
* torchvision 0.2.1
* scikit-learn 0.19.2
* scikit-image 0.14.0

and data processing lib including

- google-cloud-storage 1.10.0
- google-cloud-vision 0.33.0
- boto3 1.7.4
- joblib 0.11

## Usage
### Prepare
Run mongoDB and load database `invoice`
### Data collection and dataset creation
```bash
mkdir ~/workspace
cd ~/workspace
git clone https://github.com/Gandor26/invoice.git
cd invoice
python prepare.py [vhost name] --download --ocr --split --create
```
Note that there might be a delay while downloading OCR json files from Google cloud after they are generated.
Once finished, a random split of training/test dataset will be available in `~/workspace/invoice/data/set`
### Train model
Simply running

```bash
python main.py data/set
```

will load default configurations and start training.

Running

```bash
python main.py --help
```

will show a list of argument to configure the model. 

You can press Ctrl-C anytime to stop training and start doing test on the pre-defined test set with the best model evaluated on a randomly generated validation set from training set.

All the output will be available in `~/workspace/invoice/logs/{train,test}.log`.



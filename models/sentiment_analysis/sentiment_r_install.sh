wget https://github.com/trinker/sentimentr/tarball/master
tar -xvf master
rm -rf master

python3 -m pip install --upgrade setuptools
sudo apt-get update
sudo apt-get install python3-dev r-base r-base-dev
pip install -U pip setuptools 
pip install rpy2 pandas
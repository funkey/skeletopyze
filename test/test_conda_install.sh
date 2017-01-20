#!/bin/bash

for version in 2.7 3.5
do

  conda env remove -n skeletopyze_test_conda_install -y

  conda create -n skeletopyze_test_conda_install python=${version} -y
  source activate skeletopyze_test_conda_install

  conda config --add channels ukoethe
  conda install -c funkey skeletopyze -y

  python -c 'import skeletopyze'
  if [ $? == 0 ]
  then \
    echo "skeletopyze successfully installed"
  else \
    echo "import failed!"
    exit 1
  fi

  source deactivate
  conda env remove -n skeletopyze_test_conda_install -y

done

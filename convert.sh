#!/usr/bin/bash
case "$1" in
  "from")
    echo "convert from .py to .ipynb"
    ipynb-py-convert "./classification.py" "./classification.ipynb"
    ipynb-py-convert "./regression.py" "./regression.ipynb"
    ipynb-py-convert "./test.py" "./test.ipynb"
    ;;
  "to")
    echo "convert from .ipynb to .py"
    ipynb-py-convert "./classification.ipynb" "./classification.py"
    ipynb-py-convert "./regression.ipynb" "./regression.py"
    ipynb-py-convert "./test.ipynb" "./test.py"
    ;;
  "rm")
    rm ./*.ipynb;;
  "*")
    echo "Invalid Command"
esac


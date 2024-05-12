### 1. Usage
+ Prepare the data:
    - Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/), [WHU](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html), and [SYSU](https://github.com/liumency/SYSU-CD)
    - Generate list file as `ls -R ./label/* > test.txt`
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```
    
+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n PGPANet python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `

=======
prepare the data:

Download datasets LEVIR, BCDD, and SYSU
Prepare datasets into the following structure and set their path in train.py and test.py
├─Train
    ├─A        ...jpg/png
    ├─B        ...jpg/png
    ├─label    ...jpg/png
    └─list     ...txt
├─Val
    ├─A
    ├─B
    ├─label
    └─list
├─Test
    ├─A
    ├─B
    ├─label
    └─list
Prerequisites for Python:

Creating a virtual environment in the terminal: conda create -n A2Net python=3.8
Installing necessary packages: pip install -r requirements.txt 

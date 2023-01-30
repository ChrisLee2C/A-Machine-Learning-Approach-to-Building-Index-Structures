1.Installation procedures
install the following packages, make sure the version of python and tensorflow are the same:
python==3.7
numpy==1.21.6
pwlf==2.2.1
scikit_learn==1.1.1
scipy==1.7.3
tensorflow==1.15.0
pip is recommended for the installation

2.Operating instructions
2.1 Modify the following parameters in main.py:
To generate normal distributed data, modify  "datatype.nor_gen(low, high, size)" to "nor_gen(mean, sigma, size)", and input the mean, sigma, size
To generate uniform distributed data, input low, high, size
2.2 Run main.py
2.3 Runtime and comparison .txt files will be generated for further analysation
import os

for root, dirs, files in os.walk("./Images"):
    c = 0
    for filename in files:
        c += 1
        #if c == 18:
        print(filename)
        parameters = "python main.py 5 256 256 " + filename + " 1.0 1.0 1 0 0 100000"
        os.system(parameters)
        #    break


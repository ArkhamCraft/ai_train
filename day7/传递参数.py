import sys
print(sys.argv)
file = open(sys.argv[1],encoding="utf8")
print(file.read())
file.close()
file = open(sys.argv[2],encoding="utf8")
print(file.read())
file.close()
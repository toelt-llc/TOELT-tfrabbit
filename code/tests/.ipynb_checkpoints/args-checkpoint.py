#!/usr/bin/python


import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print ('Input file is :', inputfile)
   print ('Output file is :', outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])

def main(argv):
   neurons = ''
   predictions = ''
   try:
      opts, args = getopt.getopt(argv,"hn:p:",["neurons=","predictions="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Usage : args.py -n <neurons> p <npredictions>')
         print('Default neurons array is [5, 10, 100, 500], predictions to compute is 50000')
         sys.exit()
      elif opt in ("-n", "--neurons"):
         neurons = arg
      elif opt in ("-p", "--npredictions"):
         predictions = arg
   print ('Neurons array is :', neurons)
   print ('Prediction is :', prediction)

if __name__ == "__main__":
   main(sys.argv[1:])
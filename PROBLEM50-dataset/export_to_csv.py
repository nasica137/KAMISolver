import csv

def convert_from_txt_to_csv(inFile, parameter):
    outFile = inFile.split(".txt")[0] + ".csv"
    with open(outFile, 'w') as csvfile:
        
        csvfile.write('level;%s\n' % (parameter))
        
        with open(inFile, 'r') as file:
            for number, line in enumerate(file):
                if len(line) != 0:
                    data = str(number) + ";" + str(line)
                    csvfile.write(data)
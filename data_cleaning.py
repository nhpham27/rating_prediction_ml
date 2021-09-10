# cleaning the data
import csv
import langid
from langdetect import detect
from numpy import *
from standardize_quote import standardize_quote

# data file name
fname = "data/data.csv"

# initialize the title and row list
fields = []
rows = []

# read the csv file
with open(fname, 'r', encoding="utf-8") as in_file, open("data/cleaned_data.csv", 'w', encoding="utf-8", newline='') as out_file:
    # creating csv file reader and writer object
    csv_reader = csv.reader(in_file)
    csv_writer = csv.writer(out_file)
    
    # extracting field names in the first row of input file
    fields = next(csv_reader)

    # write field names to output file
    csv_writer.writerow(fields)

    # add all rows to the array
    for row in csv_reader:
        rows.append(row)

    # initialize counters
    en_counter = 0
    both_counter = 0
    all_counter = 0
    loop_counter = 1

    # loop through all rows of input file move rows with 
    # adequate data and english language to output file
    num_row = len(rows)
    for row in rows:
        print("Processing row: " + str(loop_counter) + "/" + str(num_row), end="\r")
        loop_counter += 1
        if len(row) >= 3:
            all_counter += 1
            # # check language of the data using langid library
            lang = langid.classify(row[0])
            if lang[0] == 'en':
                try:
                    # detect the language again using langdetect library
                    if detect(row[0]) == 'en':
                        both_counter += 1
                        temp = row.copy()
                        # standardize the quotation mark, replace '’' with "'"
                        temp[0] = standardize_quote(temp[0])

                        # write the row to the file
                        csv_writer.writerow(temp)
                except:
                    both_counter = both_counter
                en_counter += 1

    # print the statistics
    print("Processing row: " + str(loop_counter) + "/" + str(num_row))
    print("English data rows detected:")
    print("langid = " + str(en_counter) + "/" + str(all_counter))
    print("langid + langdetect = " + str(both_counter) + "/" + str(all_counter))
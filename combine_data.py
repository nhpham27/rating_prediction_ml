# combine all the csv data files generated from scraping the data 

import os

# open links file to count number of links available
with open("data/data.csv", 'w', encoding="utf-8") as out_file:
    # open output file for writing merged data
    with open("links.txt", 'r') as in_file:
        lines = in_file.readlines()
        num_file = len(lines)
        # open all data files and merge them into ouput file
        for i in range(1, num_file):
            fname = "data" + str(i) + ".csv"
            # open data file for reading
            with open(fname, 'r', encoding="utf-8") as f:
                rows = f.readlines() # read lines
                # write lines to output file
                for row in rows:
                    out_file.writelines(row)
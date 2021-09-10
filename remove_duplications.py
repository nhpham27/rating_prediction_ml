import os
import csv

# remove the duplications from the data
with open("data/cleaned_data.csv", 'r', encoding="utf-8") as in_file:
    csv_reader = csv.reader(in_file)
    
    rows = []
    titles = next(csv_reader)
    for row in csv_reader:
        rows.append(row)

    test_list = []
    indices = []
    for i in range(0, len(rows) - 1):
        print("i = " + str(i))
        if rows[i][0] not in test_list:
            test_list.append(rows[i][0])
            indices.append(i)
    with open("data/completed_data.csv", 'w', encoding="utf-8", newline='') as out_file:
        csv_writer = csv.writer(out_file)
        
        for index in indices:
            csv_writer.writerow(rows[index])
        
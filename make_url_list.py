import os
import csv


install_list = [os.listdir("install_files")]
f = open('tt.csv','w', newline='')
wr = csv.writer(f)
for file in install_list:
    filename = file.split("-")[0]
    version = file.split("-")[1]    
    wr.writerow([f"install_files/{file}", f"https://pypi.org/project/{filename}/{version}/#files"])
f.close()
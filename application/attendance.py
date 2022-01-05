from commonfunctions import *


def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',') #.csv is short for comma separated values
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
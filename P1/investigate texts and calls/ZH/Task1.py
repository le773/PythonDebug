"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv

contacts = set()

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

for line in texts:
    # print(line[0])
    # print(line[1])
    # if line[0] not in contacts:
    contacts.add(line[0])
    # if line[1] not in contacts:
    contacts.add(line[1])


with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

for line in calls:
    # pass
    # if line[0] not in contacts:
    contacts.add(line[0])
    # if line[1] not in contacts:
    contacts.add(line[1])

# print(contacts)
print('There are %d different telephone numbers in the records.' % len(contacts))

total = 0
# for line in contacts:
#     if line is not None:
#         total += 1

'''
# check phone numbers
phonenumbers = {}
for line in contacts:
    # print(line)
    if line not in phonenumbers:
        phonenumbers[line] = 1
    else:
        phonenumbers[line] = phonenumbers[line] + 1

for key in phonenumbers:
    if phonenumbers[key] > 1:
        print("%s %s" %(key, phonenumbers[key]))
'''

"""
任务1：
短信和通话记录中一共有多少电话号码？每个号码只统计一次。
输出信息：
"There are <count> different telephone numbers in the records.
"""

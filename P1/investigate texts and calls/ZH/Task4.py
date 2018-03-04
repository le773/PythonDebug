"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

msger = set()

for line in texts:
    # print(line[0], line[1], line[2])
    msger.add(line[0])
    msger.add(line[1])


with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

# answer set
answers = set()

for line in calls:
    answer = line[1]
    if answer not in answers:
        answers.add(answer)

# 电话促销员
telemarketers = set()

for line in calls:
    caller = line[0]
    # if caller.startswith('140') is False:
    #     continue
    if caller in answers:
        continue
    if caller in msger:
        continue
    telemarketers.add(caller)


print("These numbers could be telemarketers: ")
telemarketers_sort = sorted(telemarketers)
for telemarketer in telemarketers_sort:
    print(telemarketer)

"""
电话促销员的号码没有括号或空格 , 但以140开头。

任务4:
电话公司希望辨认出可能正在用于进行电话推销的电话号码。
找出所有可能的电话推销员:
这样的电话总是向其他人拨出电话，
但从来不发短信、接收短信或是收到来电


请输出如下内容
"These numbers could be telemarketers: "
<list of numbers>
电话号码不能重复，每行打印一条，按字典顺序排序后输出。
"""


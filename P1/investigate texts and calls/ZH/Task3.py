"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv
import re

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


bangaloreCaller = []
bangaloreAnswer = 0
for line in calls:
    caller = line[0]
    answer = line[1]
    if caller.strip().startswith('(080)'):
        bangaloreCaller.append(answer)
    else:
        continue
    if answer.strip().startswith('(080)'):
        bangaloreAnswer += 1

# 所有电话的区号和移动前缀
bangaloreCallerSimple = set()  # remove duplicate scripts
for line in bangaloreCaller:
    # refs: https://segmentfault.com/q/1010000010176540
    # tmp = re.findall(r'[^()]+', line)[0]
    # if tmp is not None:
    #     print('---->', tmp)
    #     bangaloreCallerSimple.add(tmp)
    #     continue
    if line.startswith('('):
        left = line.index('(') + 1 # start '( + 1'
        right = line.index(')')
        tmp = line[left:right]
        # print(line, tmp)
        bangaloreCallerSimple.add(tmp)
        continue
    bangaloreCallerSimple.add(line[0:4])


bangaloreCaller_sort = sorted(bangaloreCallerSimple)
print("The numbers called by people in Bangalore have codes: ")
# print(bangaloreCaller_sort)
for line in bangaloreCaller_sort:
    print(line)

# number of caller is bangalore
callerSize = len(bangaloreCaller)

# (number of answer is bangalore) / (number of caller is bangalore)
bangaloreCallerAnswer = bangaloreAnswer/callerSize * 100

# print(bangaloreAnswer,callerSize, bangaloreCallerAnswer)

print('%.2f percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore.' % bangaloreCallerAnswer)
"""
任务3:
(080)是班加罗尔的固定电话区号。
固定电话号码包含括号，
所以班加罗尔地区的电话号码的格式为(080)xxxxxxx。

第一部分: 找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀（代号）。
 - 固定电话以括号内的区号开始。区号的长度不定，但总是以 0 打头。
 - 移动电话没有括号，但数字中间添加了
   一个空格，以增加可读性。一个移动电话的移动前缀指的是他的前四个
   数字，并且以7,8或9开头。
 - 电话促销员的号码没有括号或空格 , 但以140开头。

输出信息:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
代号不能重复，每行打印一条，按字典顺序排序后输出。

第二部分: 由班加罗尔固话打往班加罗尔的电话所占比例是多少？
换句话说，所有由（080）开头的号码拨出的通话中，
打往由（080）开头的号码所占的比例是多少？

输出信息:
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
注意：百分比应包含2位小数。
"""

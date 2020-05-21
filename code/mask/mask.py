with open('t3.txt', 'r') as file:
    input_lines = [line.strip("\n") for line in file]

temp=[]
for l in input_lines:
    a=[]
    l=l.replace(" ","")
    for i in l:
        a.append(int(i))
    temp.append(a)

# print(temp)
# for i in temp:
    # print(*i)

for i in range(len(temp)):
    l=temp[i]
    fl=0
    for j in range(len(l)):
        if l[j] == 2:
            fl=1
        if fl and l[j] == 1:
            temp[i][j]=2



# """
f=open("t4.txt",'a')
for i in range(121):
    s1=[str(i) for i in temp[i]]
    s2=" ".join(s1)
    f.write(s2)
    f.write("\n")
f.close()
# """
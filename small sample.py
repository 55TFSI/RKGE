import random

if __name__ == '__main__':

    name = 'data/training_all.txt'
    with open(name, 'r', encoding='utf-8') as f:
        lines = f.readlines()#获取所有行
        sum = 0
        list = []
        for line in lines:#第i行

            list.append(line)


    with open('data/traning_for_parameter.txt', 'w', encoding='utf-8') as g:
        a = random.sample(list, 12000)#随机抽取500行
        for i in a:
            g.write(i)
    f.close()
    g.close()
    print(sum)
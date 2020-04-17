
if __name__ == '__main__':

    f=open('/home/xiao/PycharmProjects/毕业设计/data/flight_2008.csv','r')
    first_line=f.readline()
    print(first_line)
    f1=open('/home/xiao/PycharmProjects/毕业设计/data/train_sample_flight.csv','w')
    f2=open('/home/xiao/PycharmProjects/毕业设计/data/test_sample_flight.csv','w')
    f1.write(first_line)
    f2.write(first_line)
    i=0
    for line in f.readlines():
        i+=1
        if i<=200000:
            f1.write(line)
        elif i<=240000:
            f2.write(line)
        else:
            break




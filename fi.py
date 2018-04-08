'''fp = open("/home/venky/Downloads/dataset_.csv","r")
fp1 = open("/home/venky/Downloads/dataset_1.csv","w")


lines = fp.readlines()
blob = []
for line in lines:
    blob.append(",".join([i.strip() for i in line.split(',')]))

print blob


for item in blob:
    fp1.write("%s\n" % item)

fp.close()
'''
import socket
global count
def host(ip):
    try:
        p = socket.gethostbyaddr(ip)
        str = p[0]
    except :
        str = "Unk"
    return str


fp = open("/home/venky/Downloads/processed_merged.csv",'r')
lines = fp.readlines()
p = []
for line in lines:
    x,y,z = line.split(",")
    p.append(host(x))

print len(p)

p[:] = [x for x in p if x != "Unk"]

print len(p)

print set(p)
print len(set(p))

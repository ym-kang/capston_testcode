



def log_parse():
    data_arr = []
    with open("log/training.log") as f:
        content = f.readlines()

    
    for i,dat in enumerate(content):
        if(dat!="\n"):
            data_arr.append(dat)
    import re
    data_arr2 = []
    counter = 0
    for dat in data_arr:
        
        content = dat.replace("\n","").replace("\t"," ")
        content = content.split(" ")
        if("detector.c:142:" in content):
            time = content[1]
            counter = 1
            avg_loss = content[6]
        elif("detector.c:591:" in content):
            if(counter==1):
                training_set_iou = content[22]
            else:
                test_set_iou = content[22]
            counter+=1
            
        if(counter==3):
            data_arr2.append((time,avg_loss,training_set_iou,test_set_iou))

            pass


    return data_arr2


'''
[0]:'09:58:52'
[1]:'26.271412'
[2]:'14.57%'
[3]:'5.65%'
'''   

result = log_parse()

a = 1
import matplotlib.pyplot as plot

def show_graph(log_data):

    x = [i for i in range(len(log_data))]
    y = [data[1] for data in log_data]
    y2 =[float(data[2].replace("%","")) for data in log_data]
    y3 =[float(data[3].replace("%","")) for data in log_data]
    plot.subplot(211)
    plot.plot(x ,y )
    plot.subplot(212)
    plot.plot(x,y2,"r-",x,y3,"b-")
    
    plot.show()

    pass


show_graph(result)
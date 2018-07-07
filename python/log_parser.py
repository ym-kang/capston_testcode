



def log_parse():
    data_arr = []
    with open("training.log") as f:
        content = f.readlines()

    
    for i,dat in enumerate(content):
        if(dat!="\n"):
            data_arr.append(dat)
    import re
    data_arr2 = []
    counter = 0
    for dat in data_arr:
        
        content = dat.replace("\n","").replace("\t"," ")
        content = content.split()
        #content =" ".join(content)
        if("detector.c:142:" in content):
            time = content[1]
            counter = 1
            avg_loss = content[6]
        elif("detector.c:591:" in content):
            try:
                if(counter==1):
                    training_set_iou = content[10]
                    #val = training_set_iou.replace("%","")
                    #val = float(val)
                else:
                    test_set_iou = content[10]
                    #val = training_set_iou.replace("%","")
                    #val = float(val)
            except:
                counter = 0
                continue

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



left  = 0.125  # the left side of the subplots of the figure
right = 0.7    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots

import matplotlib.pyplot as plt

def show_graph(log_data):

    x = [i*100 for i in range(len(log_data))]
    y = [float(data[1]) for data in log_data]
    y2 =[float(data[2].replace("%","")) for data in log_data]
    y3 =[float(data[3].replace("%","")) for data in log_data]
    #fig, axes = plot.subplots(2,1)
    #fig.tight_layout()
    plt.subplot(211)
    plt.plot(x ,y )
    plt.title("Loss Decay")
    plt.xlabel("iteration")
    plt.ylabel("avg loss")
    axes = plt.gca()
    #axes.set_xlim([0,40100])



    plt.subplot(212) 
    test1, = plt.plot(x,y2,"r-",label="Train Set")
    test2, = plt.plot(x,y3,"b-",label="Test Set")
    plt.legend(handles=[test1,test2])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title("Train&Test Set IOU")
    plt.xlabel("iteration")
    plt.ylabel("IOU")
    
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    axes = plt.gca()
    #axes.set_xlim([0,40100])
    plt.show()

    pass

show_graph(result)
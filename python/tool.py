import time

def checkTime(tag):
    if not hasattr(checkTime,'start'):
        checkTime.start = time.time()
    if not hasattr(checkTime,'printTime'):
        checkTime.printTime = False
    
    elapsed = time.time()-checkTime.start
    
    if checkTime.printTime:
        print(tag,"elapsed:",elapsed)
    checkTime.start = time.time()
    return elapsed
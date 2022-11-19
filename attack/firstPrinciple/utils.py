import json
def jload(filepath):
    f = open(filepath)
    tmp = json.load(f)
    f.close()
    return tmp

def jdump(obj, filepath):
    f = open(filepath,'w')
    json.dump(obj,f)
    f.close()
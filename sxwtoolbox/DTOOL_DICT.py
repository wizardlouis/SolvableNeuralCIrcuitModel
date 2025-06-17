# dict operation
def dunion(a:dict,b:dict) -> dict:
    union=a.copy()
    for key,value in b.items():
        union[key]=value
    return union

def dintersect(a:dict,b:dict)->dict:
    intersect={}
    for key in a.keys() & b.keys():
        intersect[key]=b[key]
    return intersect




# Class: Mydict
class Mydict:

    def __init__(self, origindict=None, **kwargs):
        if origindict is not None:
            assert type(origindict) == dict
            for key in origindict.keys():
                self.__dict__[key] = origindict[key]
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]

    def __setattr__(self, item, value):
        self.__dict__[item] = value

    def __getattribute__(self, item):
        return super().__getattribute__(item)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return super().__getattribute__(key)

    def get(self,key,value=None):
        return self.__dict__.get(key,value)

    def show(self):
        return self.__dict__

    def __add__(self, other):
        assert type(other) == Mydict
        return Mydict(dict(**self.__dict__, **other.__dict__))

    def todict(self):
        return self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def element_add(self,other):
        assert type(other) == Mydict
        selfkey=self.keys()
        otherkey=other.keys()
        intersection=selfkey & otherkey
        selfdiff=selfkey - intersection
        otherdiff=otherkey - intersection
        result = Mydict()
        for key in intersection:
            result[key] = self[key]+other[key]
        for key in selfdiff:
            result[key] = self[key]
        for key in otherdiff:
            result[key] = other[key]
        return result

    def element_append(self,other):
        assert type(other) == Mydict
        selfkey=self.keys()
        otherkey=other.keys()
        intersection=selfkey & otherkey
        selfdiff=selfkey - intersection
        otherdiff=otherkey - intersection
        result = Mydict()
        for key in intersection:
            result[key]=self[key]+[other[key]]
        for key in selfdiff:
            result[key]=self[key]
        for key in otherdiff:
            result[key]=[other[key]]
        return result

    def coverage(self,object):
        #coverage contents with another object Mydict or dict file
        for key in object.keys():
            self.__dict__[key] = object.__dict__[key]
        return self

    def search(self,keylist):
        result=Mydict()
        for key in keylist:
            if key in self.keys():
                result[key]=self[key]
            else:
                raise KeyError(f'In Mydict.search() method, Mydict object does not include {key}!!!')
        return result

    def update(self,datadict):
        for key in datadict.keys():
            self.__setattr__(key,datadict[key])

# if __name__ == '__main__':
#     A=Mydict(a=1,b=2)
#     print(A.get('a'))
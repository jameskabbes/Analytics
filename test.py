


def one(a,b, **kwargs):

    print (a)
    print (b)

    for i in kwargs:

        print (i, kwargs[i])


def main():

    set_one = dict( yes = 1, whoa = 2 )

    one(1,2, asdf= 'poop', yeet= 'mhm', **set_one)








main()

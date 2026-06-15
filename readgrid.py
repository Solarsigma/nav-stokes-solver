#/usr/bin/python3

def readgrid(filepath):

    '''
    this is a very hacky way of gathering the coordinates - initialize
    empty arrays and populate them dynamically by appending
    coordinate information. Removes the need to know array size
    info apriori, but is usually bad for big files (~50K+ lines)
    '''

    # again we use 1D arrays because they are slightly more efficient, but 
    # it's easy enough to make them 2D
    x = []
    y = []

    # open file and read line by line to assign to array
    with open(filepath,"r") as fp:
        [nx, ny] = [int(m) for m in fp.readline().strip("\n").split(", ")]
        for line in fp:
            coords = line.strip("\n").split(", ")
            x.append(float(coords[0]))
            y.append(float(coords[1]))

    # note you can also use meshgrid(x,y) to create a mesh out of this!

    # write out coordinates again
    # with open("g33x25u_py.dat","w") as fout:
    #     fout.write("\t ZONE\t i = %d, j = %d\n" %(nx, ny,))
    #     for (xx,yy) in zip(x,y):
    #         fout.write("\t%s,\t%s\n" %(xx,yy,))

    # when doing file I/O using "with open ..." you don't need to do file.close()
    # python automatically closes the file once the I/O processes are finished

    return (nx,ny), (x,y)

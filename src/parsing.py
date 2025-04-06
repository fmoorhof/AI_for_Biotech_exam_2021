import numpy as np

def parsing(txpath, data):
    # pythonic file parsing because of strings
    y_id = []
    y = []
    file = open(txpath, "r")
    for line in file:
        ys = line.strip().split(",")
        y_id.append(ys[0])
        y.append(
            ys[1].replace('NA', '2'))
    file.close()
    X_id = []
    X = []
    file = open(data, "r")
    for line in file:
        ys = line.strip().split(",")
        X_id.append(ys[0])
        X.append(ys[1:])  # could i parse as float directly? then i would get rid of fking strings; oder parsing mit if abfrage um headder raus. wird aber langsamer?
    file.close()

    # as np.array() -> this is maybe not needed now
    y_id = np.array(y_id[1:])
    X_id = np.array(X_id[1:])
    y_head = np.array(y[0])
    X_head = np.array(X[0])
    # adapt data types to less memory consumption convert horrorble little engine unicode (dtype('<U12')) into floats
    X = np.array(X[1:], dtype='float64')
    y = np.array(y[1:], dtype='i1')  # int8
    # X = np.frompyfunc(lambda l: l.replace(',',''),1,1)(X[1:]).astype('float64')#maybe float16 possible if 1st col gets removed
    # y = np.frompyfunc(lambda l: l.replace(',',''),1,1)(arr[1:]).astype('i1')
    return X, y, X_id, y_id, X_head, y_head

def stats(X, y):
    print('Number of DATA samples: ', X.shape[0])
    print('Number of DATA features: ' + str(X.shape[1]))
    print('Number of LABEL samples in total:\t' + str(y.shape[0]))
    lab, freq = np.unique(y, return_counts=True)
    print('Number of values with missing label NA:\t%d' % freq[2])
    print('Number of values with label ' + str(lab[0]) + ':\t%.2f' % freq[0])
    print('Number of values with label  %s:\t%.2f' % (lab[1], freq[1]))
    print('Percentage of values of label ' + str(lab[0]) + ':\t%.2f' % (freq[0] / y.size * 100))
    print('Percentage of values in label %s:\t%.2f' % (lab[1], (freq[1] / y.size * 100)))
    print('Percentage of values with missing labels:\t%.2f' % ((freq[2] / y.size * 100)))
    # preprocessing block gets invoked:
    print('feature\tzeros abs.\tzeros %')#uncomment in filtering loop for data printing



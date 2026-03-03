import math


def softmax(arr):
    total = 0
    res = []

    for item in arr:
        total += math.exp(item)

    for item in arr:
        res.append(item / total)

    return res

if __name__ == '__main__':
    print(softmax([29, 40, 121, 64]))
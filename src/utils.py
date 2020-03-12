import matplotlib.pyplot as plt


# class PriorityQueue(object):

#     def __init__(self):
#         self.queue = []
    
#     def get(self):
#         return self.queue.pop()
    
#     def put(self, item):
#         self.queue.append(item)
#         for i in range(1, len(self.queue)):
#             if self.queue[i][0] > self.queue[i - 1][0]:
#                 temp = self.queue[i - 1]
#                 self.queue[i - 1] = self.queue[i]
#                 self.queue[i] = temp
#             else:
#                 break

    
#     def remove(self, item):
#         for i, (prio, item) in self.queue:
#             if item == item:


# end

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x: x[1])[0]

def plot_ccs(S):
    print(S)
    f, ax = plt.subplots()
    for V_PI in S:
        x_vals = [V_PI.start.x, V_PI.end.x]
        y_vals = [V_PI.start.y, V_PI.end.y]        
        ax.plot(x_vals, y_vals)
    
    # ax.set_title("CCS approximated by OLS")
    ax.set_xlabel("w1")
    ax.set_ylabel("Vw")    
    plt.show()

def plot_ccs_2(S):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    for V_PI in S:
        x_vals = [V_PI.start.x, V_PI.end.x]
        y_vals = [V_PI.start.y, V_PI.end.y]        
        ax[0].plot(x_vals, y_vals)
        ax[1].scatter(V_PI.obj2, V_PI.obj1)
    
    # ax.set_title("CCS approximated by OLS")
    ax[0].set_xlabel("w1")
    ax[0].set_ylabel("Vw")    

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Treasure")

    plt.show()


# -*- coding: utf-8 -*-
# plot two list

import matplotlib.pyplot as plt
import matplotlib
from pylab import savefig
from scipy.interpolate import spline
import numpy as np
import os
import os.path
matplotlib.pyplot.switch_backend('agg')

def plot_lists(list1, list2, l1name=None, l2name=None, title=None, \
               xlabel=None, ylabel=None, filename=None):
    size1 = len(list1)
    size2 = len(list2)
    if(size1 != size2):
        print('size of list 1 and 2 is not equal')
        return
    x = np.arange(1, size1+1)
    if(l1name != None and l2name != None and title != None \
       and xlabel != None and ylabel != None):
        plt.plot(x, list1, 'r--', label=l1name)
        plt.xlim(left=0)
        plt.plot(x, list2, 'g--', label=l2name)
        plt.xlim(1, size1+1)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    else:
        plt.plot(x, list1,'r-', x, list2, 'g-')
    if filename!=None:
        if os.path.exists(filename + ".png"):
            os.remove(filename + ".png")
        savefig(filename)
    else:
        plt.show()
    plt.close("all")

def plot_lists_xy(x, y, xname=None, yname=None, title=None, \
               xlabel=None, ylabel=None, filename=None):
    plt.plot(x, y, 'b--', label=yname)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename!=None:
        if os.path.exists(filename + ".png"):
            print('exist')
            os.remove(filename + ".png")
        print('not exist')

        savefig(filename)
    else:
        plt.show()
    plt.close("all")

def plot_scatter_xy(x, y, title=None, \
               xlabel=None, ylabel=None, filename=None):
    plt.close("all")
    plt.scatter(x, y, s=20, c='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename!=None:
        if os.path.exists(filename + ".png"):
            print('exist')
            os.remove(filename + ".png")
        print('not exist')
        savefig(filename)
    else:
        plt.show()
    plt.close("all")

def plot_single_list(list1, l1name, title=None, xlabel=None, ylabel=None, filename=None):
    plt.close("all")
    size1 = len(list1)
    x = np.arange(1, size1+1)
    plt.plot(x, list1, 'b-,', label=l1name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename!=None:
        if os.path.exists(filename + ".png"):
            print('exist')
            os.remove(filename + ".png")
        print('not exist')
        savefig(filename)
    else:
        plt.show()
    plt.close("all")

def plot_single_list_same(list1, l1name, \
                          color='b',title=None, xlabel=None, ylabel=None, filename=None):
    size1 = len(list1)
    x = np.arange(1, size1+1)
    plt.plot(x, list1, color+'-,', label=l1name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    savefig(filename)



def annot_max(x,y, ax=None):
    ymax = min(y)
    xmax = y.index(ymax)+1
    text= "Times={:.0f}, Min Validation={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

class test(object):
    def __init__(self):
        pass

def plot_lists_as_singles(list, x_num,batch_name):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    y_num = int((len(list))/x_num)
    for i in range(y_num):
        for j in range(x_num):
            index = j + i*x_num
            value = list[index]
            if j == 0:
                x1.append(value)
            elif j == 1:
                x2.append(value)
            elif j == 2:
                x3.append(value)
            elif j == 3:
                x4.append(value)
            elif j == 4:
                x5.append(value)
            elif j == 5:
                x6.append(-value)
    i_str = 'input_dropout'
    u_str = 'update_dropout'
    o_str = 'output_dropout'
    lr_str = 'learning rate'
    lrdecay_str = 'lr_decay'
    validation_str = 'Perplexity'
    bo_list_str(x1, i_str, batch_name)
    bo_list_str(x5, u_str, batch_name)
    bo_list_str(x3, o_str, batch_name)
    bo_list_str(x4, lr_str, batch_name)
    bo_list_str(x2, lrdecay_str, batch_name)
    bo_list_str(x6, validation_str, batch_name)


def bo_list_str(list, str_name, batch_name):
      plot_single_list(list,title=str_name, xlabel= 'Bayesian Optimisation running times'\
                     , ylabel=str_name, filename='pictures/' + batch_name+ 'BO' + str_name)

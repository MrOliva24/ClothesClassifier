import random
import time
from utils_data import *
from Kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def visualitzacio(image, n_clusters):
    data = image 
    
    Km = KMeans(data, n_clusters)
    Km._init_centroids()
    Km.get_labels()
    Km.fit()
    
    plt.scatter(Km.X[:,0], Km.X[:,1], c=Km.labels,s=1 ,cmap='plasma_r')
    plt.scatter(Km.centroids[:,0], Km.centroids[:,1], color='k', s=5)
    plt.show(block=True)
    
def visualitzacio3D(image, n_clusters):
    data = image
    Km = KMeans(data, n_clusters)
    Km._init_centroids()
    Km.get_labels()
    Km.fit()
    Plot3DCloud(Km,1,1,1)
    
def Retrieval_by_color(imageDT, colors, prompt, n, show_images = True):
    
    '''
    Returns a list of indexes of the images that contain the color specified in the prompt.
    Args:
        imageDT : image dataset
        colors : list containing the color of the images calculated by Kmeans
        prompt : color to search for
        n : number of images to retrieve (-1 to show all)
        show_images : boolean to show the images or not
    '''
    idx = [i for i in range(len(colors)) if prompt in colors[i]]
    if n > len(idx):
        print(f'Asked for more images than available, showing all ({n}) images')
        n = len(idx) 
    if n == -1:
        n = len(idx)
    if show_images:
        random.shuffle(idx)
        images = imageDT[idx]
        visualize_retrieval(images, n)
    return idx

def Retrieval_by_shape(imageDT, labels, prompt, n, show_images = True):
    
    '''
    Returns a list of indexes of the images that contain the label specified in the prompt.
    Args:
        imageDT : image dataset
        labels : list containing the labels of the images calculated by KNN
        prompt : label to search for
        n : number of images to retrieve (-1 to show all)
        show_images : boolean to show the images or not
    '''
    idx = [i for i in range(len(labels)) if prompt in labels[i]]
    if n > len(idx):
        print(f'Asked for more images than available, showing all ({n}) images')
        n = len(idx) 
    if n == -1:
        n = len(idx)
    if show_images:
        random.shuffle(idx)
        images = imageDT[idx]
        visualize_retrieval(images, n, info = None)
    return idx

def Retrieval_combined(imageDT, colors, labels, prompt, n, show_images = True):
    
    '''
    Returns a list of indexes of the images that contain the color and label specified in the prompt
    Args:
        imageDT : image dataset
        labels : list containing the labels of the images calculated by KNN
        prompt : label to search for (format: 'color label')
        n : number of images to retrieve (-1 to show all)
        show_images : boolean to show the images or not
    '''
    prompt = prompt.split(' ')
    idx = [i for i in range(len(colors)) if prompt[0] in colors[i] and prompt[1] in labels[i]]
    if n > len(idx):
        print(f'Asked for more images than available, showing all ({n}) images')
        n = len(idx) 
    if n == -1:
        n = len(idx)
    if show_images:
        random.shuffle(idx)
        images = imageDT[idx]
        visualize_retrieval(images, n)
    return idx

def Get_shape_accuracy(ground_truth, predictions):
    l = len(ground_truth)
    
    if l != len(predictions):
        raise ValueError('The length of the ground truth and the predictions must be the same')
    e = np.sum(ground_truth != predictions)
    return (l-e)/l

def IoUindex(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def Get_color_accuracy(ground_truth, predictions):
    '''
    l = len(ground_truth)
    if l != len(predictions):
        raise ValueError('The length of the ground truth and the predictions must be the same')
    
    
    IoU = [ IoUindex(set(truth), set(label)) for truth, label in zip(ground_truth, predictions)]
    return np.mean(IoU) 
    '''
    l=len(ground_truth)
    if l!=len(predictions):
        print("Vectors have different lengths")
        return
    tacc=0
    for i in range(l):
        truth = ground_truth[i]
        labels = predictions[i]
        diff = len(list(filter(("White").__ne__,set(labels) - set(truth))))
        mxlen = max(len(labels),len(truth))
        if(mxlen==0): mxlen=1
        acc = (1-diff/mxlen)*100
        tacc += acc
    return round(tacc/l,2)

def Kmean_statistics(KM : KMeans , Kmax):
    WCD_values = []
    times = []
    iters = []
    
    for k in range(2,Kmax+1):
        start = time.time()
        KM.__init__(KM.X, k)
        KM.fit()
        end = time.time()
        WCD = KM.withinClassDistance()
        times.append(end - start)
        iters.append(KM.num_iters)
        WCD_values.append(WCD)
        #print(f"K = {k} -> WCD = {WCD} -> Time = {end - start} -> Iters = {KM.num_iters}")
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(range(2,Kmax+1), WCD_values)
    plt.xlabel('K')
    plt.ylabel('WCD')
    plt.title('Within Class Distance')
    
    plt.subplot(1,3,2)
    plt.plot(range(2,Kmax+1), iters)
    plt.xlabel('K')
    plt.ylabel('Iterations')
    plt.title('Number of iterations')
    
    plt.subplot(1,3,3)
    plt.plot(range(2,Kmax+1), times)
    plt.xlabel('K')
    plt.ylabel('Time')
    plt.title('Convergence time')
    
       
    plt.tight_layout()
    plt.show()
    
     

def find_K_distribution(predicted_labels, max_K):

    histogram = [0 for _ in range(max_K)]
    for labels in predicted_labels:
        histogram[len(labels)-1] += 1
    distribution = [round(elem/len(predicted_labels),4) for elem in histogram]
    return distribution


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Function used mainly to visualize long tasks. This function does not
    impact the functionality of our code in any way, we used it mainly
    to have something to look at while waiting in the terminal.

    The original source is the following:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

    This is the original documentation for the function:
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str) █
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |'+bcolors.OKCYAN+f'{bar}'+bcolors.ENDC+f'| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    
    
def RGBtoHSL(imageDT):
    shape = imageDT.shape
    count = 0
    
    for image in range(shape[0]):
        count += 1
        printProgressBar(count, shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
        for pixelH in range(shape[1]):
            for pixelV in range(shape[2]):
                item = imageDT[image][pixelH][pixelV]
                imageDT[image][pixelH][pixelV] = .299*item[0] + .587*item[1] + .114*item[2]
    
    return imageDT


def k_fold(imageDT, k, q):
    l = len(imageDT)
    if q >= k:
        raise ValueError('q must be less than k')
    if k > 1 and k < 15:
        c = round(np.floor(l/k))
        if q == 0:
            test = imageDT[:c]
            train = imageDT[c:]
        elif q == k-1:
            test = imageDT[l-c:]
            train = imageDT[:l-c]
        else:
            test = imageDT[c*q:c*(q+1)]
            train = np.concatenate((imageDT[:c*q], imageDT[c*(q+1):]))
        return train, test
    return "Si salta este error eres un inutil"

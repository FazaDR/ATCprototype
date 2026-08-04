#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:21:48 2024

@author: Corner
"""


import numpy as np
import cv2
import random
import glob
import open3d as o3d
import time

class DBL_GNG():
    def __init__(self, feature_number, maxNodeLength, L1=0.5, L2=0.01, 
                errorNodeFactor = 0.5, newNodeFactor = 0.5):
    
        
        self.feature_number = feature_number
        self.M = maxNodeLength
        self.alpha = L1
        self.beta = L2           
        self.delta = errorNodeFactor
        self.rho = newNodeFactor
        
        self.eps = 1e-04
    
    
    def resetBatch(self):    
        self.Delta_W_1 = np.zeros_like(self.W)
        self.Delta_W_2 = np.zeros_like(self.W)
        self.A_1 = np.zeros(len(self.W))
        self.A_2 = np.zeros(len(self.W))
        self.S = np.zeros((len(self.W),len(self.W)))              
        
    def initializeDistributedNode(self, data, number_of_starting_points = 1):
        
        data = data[:,:self.feature_number].copy()
        np.random.shuffle(data)
        
        nodeList = np.empty((0,self.feature_number),dtype=np.float32) 
        edgeList = np.empty((0,2),dtype=int)

        
        #copy the data, ready for crop
        tempData = data.copy()

        #define the batch size
        batchSize = len(data) // number_of_starting_points

    
    
        for i in range(number_of_starting_points):
            idx = np.arange(len(tempData), dtype=int)
            
            #randomly select a node
            selectedIndex = np.random.choice(idx[-batchSize:])
            currentNode = tempData[selectedIndex]
            
            # insert the node into list
            nodeList = np.append(nodeList, [currentNode],axis=0)
            
            # calculate the distance from all data to the current node
    
            y2 = np.sum(np.square(tempData), axis=1)
            dot_product = 2 * np.matmul(currentNode,tempData.T)
            dist =  y2 - dot_product 
            
            
            
            #perform sorting, so now we know which is closest, with is farthest
            idx = np.argsort(dist)
            
            # select the third cloest node as neighbor, try to leave some space in between
            neighborNode = tempData[idx[2]]
            # add neighbor node into the list
            nodeList = np.append(nodeList, [neighborNode],axis=0)
            
            # connect them and add to the list
            edgeList = np.append(edgeList,[[i*2, i*2 + 1]],axis=0)
            
            #randomly select a node from the farthest nodes within the batch size
            # selectedIndex = np.random.choice(idx[-batchSize:])            
            # currentNode = tempData[selectedIndex,:2]
            
            # remove the current area, so it won't repeat in the follow search
            idx = idx[batchSize:]
            tempData = tempData[idx]
        
    
        self.W = nodeList
        self.C = edgeList
        self.E = np.zeros(len(self.W))
    
        
    
    #input Batch, Feature
    def batchLearning(self, X):
        X = X[:,:self.feature_number]
        
        # identity Matrix
        i_adj = np.eye(len(self.W))
        
        adj = np.zeros((len(self.W),len(self.W)))
        adj[self.C[:,0],self.C[:,1]] += 1
        adj[self.C[:,1],self.C[:,0]] += 1        
        adj[adj > 0] = 1        
        
        
        batchIndices = np.arange(len(X))
        
        
        #obtain Distance
        x2 = np.sum(np.square(X), axis=1)
        y2 = np.sum(np.square(self.W), axis=1)              
        dot_product = 2 * np.matmul(X, self.W.T)   
    
        dist = np.clip(np.expand_dims(x2, axis=1) + y2 - dot_product, a_min=0, a_max = None)
        dist = np.sqrt(dist  + self.eps)
    
        
                
        # get fist and second winner nodes
        tempDist = dist.copy()                
        s1 = np.argmin(tempDist,axis=1)       
        tempDist[batchIndices,s1] = 99999              
        s2 = np.argmin(tempDist,axis=1)
        
        # add error to s1
        self.E += np.sum(i_adj[s1] * dist, axis=0) * self.alpha
        
        # Update s1 position
        self.Delta_W_1 += (np.matmul(i_adj[s1].T, X) - (self.W.T * np.sum(i_adj[s1],axis=0)).T)  * self.alpha
        
        
        # Update s1 neighbor position        
        self.Delta_W_2 += (np.matmul(adj[s1].T, X) - np.multiply(self.W.T, adj[s1].sum(0)).T) * self.beta
    
        
        # Add 1 to s1 node activation
        self.A_1 += np.sum(i_adj[s1], axis=0)
                
        # Add 1 to neighbor node acitvation
        self.A_2 += np.sum(adj[s1], axis=0) 
        

        # Count the important edge (s1 and s2)
        connectedEdge = np.zeros_like(self.S)              
        connectedEdge[s1,s2] = 1
        connectedEdge[s2,s1] = 1

        t = i_adj[s1] + i_adj[s2]
        connectedEdge *= np.matmul(t.T,t)
        
        self.S += connectedEdge 
        
        
    def updateNetwork(self):
        
    
        self.W += (self.Delta_W_1.T * (1 / (self.A_1 + self.eps))).T + (self.Delta_W_2.T * (1 / (self.A_2 + self.eps))).T        
        
        self.C = np.asarray(self.S.nonzero()).T
    
        
        
        self.removeIsolatedNodes()
        
        
        self.E *= self.delta
        
        if random.random() > 0.9:
            self.removeNonActivatedNodes()
        
        
    def removeIsolatedNodes(self):
        adj = np.zeros((len(self.W),len(self.W)))
        adj[self.C[:,0],self.C[:,1]] = 1
        adj[self.C[:,1],self.C[:,0]] = 1
        
        
        isolatedNodes = (np.sum(adj, axis=0) + np.sum(adj, axis=1) == 0).nonzero()[0]
    
        finalDelete = list(np.unique(isolatedNodes))
    
        if len(finalDelete) > 1:            
            finalDelete.sort(reverse=True)   
    
        for v in finalDelete:                       
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))]
                    
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1       
            
    
        if len(finalDelete) > 0:
            print("Isolated",finalDelete)
            
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)
            
            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)
        
        
    def removeNonActivatedNodes(self):
    
        nodeActivation = self.A_1 
        nonActivatedNodes = (nodeActivation == 0).nonzero()[0]
    
        finalDelete = list(nonActivatedNodes)
        
        if len(finalDelete) > 1:                
            finalDelete.sort(reverse=True)   
    
        for v in finalDelete:    
        
            self.C = self.C[np.logical_not(np.logical_or(self.C[:,0] == v, self.C[:,1] == v))]
            
            self.C[self.C[:,0] > v,0] -= 1
            self.C[self.C[:,1] > v,1] -= 1           
            
        if len(finalDelete) > 0:
            print("Non Activated",finalDelete)
            
            
            self.S = np.delete(self.S, finalDelete,axis=0)
            self.S = np.delete(self.S, finalDelete,axis=1)
            
            self.W = np.delete(self.W, finalDelete, axis=0)
            self.E = np.delete(self.E, finalDelete, axis=0)
            self.A_1 = np.delete(self.A_1, finalDelete, axis=0)
            self.A_2 = np.delete(self.A_2, finalDelete, axis=0)

    
    def addNewNode(self):
        g = np.sum(self.E > np.quantile(gng.E,0.85))
        
        for _ in range(g):
            if len(self.W) >= self.M:
                return
            
            q1 = np.argmax(self.E)
            if self.E[q1] <= 0:          
                print("Zero error q1")
                return
            
            # get the connected nodes
            connectedNodes = np.unique(np.concatenate((self.C[self.C[:,0] == q1,1], self.C[self.C[:,1] == q1,0])))  
            if len(connectedNodes) == 0:
                return
        
            
            # get the maximum error of neighbors
            q2 = connectedNodes[np.argmax(self.E[connectedNodes])]
            if self.E[q2] <= 0:              
                print("Zero error q2")
                return
        
            
            # insert new node between q1 and q2
            q3 = len(self.W)
            new_w = (self.W[q1] + self.W[q2]) * 0.5        
            self.W = np.vstack((self.W, new_w))            
            self.E = np.concatenate((self.E,np.zeros(1)),axis=0)
            
        
            # update the error 
            self.E[q1] *= self.rho
            self.E[q2] *= self.rho        
            self.E[q3] = (self.E[q1] + self.E[q2]) * 0.5 
            
        
                    
            #remove the original edge
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q1, self.C[:,1] == q2))]
            self.C = self.C[np.logical_not(np.logical_and(self.C[:,0] == q2, self.C[:,1] == q1))]
            
        
            #add the edge
            self.C = np.vstack((self.C,np.asarray([q1,q3])))
            self.C = np.vstack((self.C,np.asarray([q2,q3])))
    
    
            #add a col and row       
            self.S = np.pad(self.S, pad_width=((0, 1), (0, 1)), mode='constant')
            
        
            self.S[q1,q2] = 0
            self.S[q2,q1] = 0
            
            self.S[q1,q3] = 1
            self.S[q3,q1] = 1
            
            self.S[q2,q3] = 1
            self.S[q3,q2] = 1  
            
        
            self.A_1 = np.concatenate((self.A_1,np.ones(1)),axis=0)  
            self.A_2 = np.concatenate((self.A_2,np.ones(1)),axis=0)  
        
    
    def cutEdge(self):
        
        self.removeNonActivatedNodes()
        
        mask = self.S > 0
        
        filterV = np.quantile(self.S[mask], 0.15) 
        
        temp = self.S.copy()
        temp[self.S <filterV] = 0
        self.C = np.asarray(temp.nonzero()).T
    
        
        self.removeIsolatedNodes()
    



def displayGraph(data, nodeList, edgeList,
                minX, maxX, minY, maxY):

    padding = 2
    img_size = 1024
    
    
    lenX = maxX - minX + padding
    lenY = maxY - minY + padding
    
    scaleX = img_size/lenX
    scaleY = img_size/lenY
    
    #Print GNG Result
    image = np.ones((img_size,img_size,3),dtype=np.uint8) * 255
    # Normal Data
    for x,y,c in data:
        image = cv2.circle(image, (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY)) , 5,  [200,200,200] , -1) 
        
    W = nodeList
    E = edgeList      

    for e in E:
        x, y = W[e[0]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        start_point = p
        x, y = W[e[1]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        end_point =  p
        image = cv2.line(image, start_point, end_point, [50,50,50],3)     
        

        

    for i, d in enumerate(W):
        
        p = [int((d[0]-minX+padding//2)*scaleX), int((d[1]-minY+padding//2)*scaleY)]
        image = cv2.circle(image, p , 7,  [50,50,50] , -1)  
        
    
        
        
    for e in E:
        x, y = W[e[0]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        start_point = p
        x, y = W[e[1]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        end_point =  p
        
        
    
    cv2.imshow("window", image)  
    
        
    cv2.waitKey(2)    
    return image


# =====================================================
# Load all scans
# =====================================================

scan_files = sorted(glob.glob("dataset/scans/scan_*.csv"))

frames = []

for file in scan_files:

    pts = np.loadtxt(
        file,
        delimiter=",",
        skiprows=5
    )

    frames.append(pts.astype(np.float32))

print("Loaded", len(frames), "frames")

# =====================================================
# Compute visualization bounds
# =====================================================

all_points = np.vstack(frames)

min_bound = all_points.min(axis=0)
max_bound = all_points.max(axis=0)

# =====================================================
# Initialize GNG using FIRST FRAME ONLY
# =====================================================

gng = DBL_GNG(
    feature_number=3,
    maxNodeLength=500
)

gng.initializeDistributedNode(
    frames[0],
    number_of_starting_points=10
)

# =====================================================
# Open3D Visualizer
# =====================================================

vis = o3d.visualization.Visualizer()
vis.create_window("DBL-GNG")

pcd = o3d.geometry.PointCloud()
nodes = o3d.geometry.PointCloud()
lines = o3d.geometry.LineSet()

vis.add_geometry(pcd)
vis.add_geometry(nodes)
vis.add_geometry(lines)

ctr = vis.get_view_control()

center = (min_bound + max_bound) / 2

ctr.set_front([0,-1,0])   # look down Y
ctr.set_up([0,0,1])        # Z to the top of screen
ctr.set_lookat(center)
ctr.set_zoom(0.35)

# =====================================================
# Process ONE FRAME AT A TIME
# =====================================================

LEARNING_ITERATIONS = 60

for frame_id, current_data in enumerate(frames):

    print("----------------------------")
    print("Frame", frame_id)
    print("Input points:", len(current_data))

    # ======================================
    # Learn this frame many times
    # ======================================

    for i in range(LEARNING_ITERATIONS):

        gng.resetBatch()
        gng.batchLearning(current_data)
        gng.updateNetwork()
        gng.addNewNode()

        vis.clear_geometries()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current_data)
        pcd.paint_uniform_color([0.8, 0.8, 0.8])

        nodes = o3d.geometry.PointCloud()
        nodes.points = o3d.utility.Vector3dVector(gng.W)
        nodes.paint_uniform_color([1, 0, 0])

        lines = o3d.geometry.LineSet()
        lines.points = o3d.utility.Vector3dVector(gng.W)
        lines.lines = o3d.utility.Vector2iVector(gng.C)

        colors = np.tile([0, 0, 1], (len(gng.C), 1))
        lines.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(pcd)
        vis.add_geometry(nodes)
        vis.add_geometry(lines)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.05)      # about 20 FPS

print("Finished")

vis.run()

vis.destroy_window()

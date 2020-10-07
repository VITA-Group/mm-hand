
# coding: utf-8

# In[ ]:


#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np

dic = json.load(open("fixed_view/label_fixed_view_2000.json","r"))
H_scale = 1920//4
W_scale = 1080//4
H_offset = 960//4
W_offset = 540//4

def return_img_path(i):
    return "fixed_view/%04d.png" % (i+1)

def plot_hand(coords_hw, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])
    # define connections and colors of the bones
    print(coords_hw[-1]) # this is center ( the 22nd point)
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]
    ret_coor = []
    for connection, color in bones:
        coord1 = coords_hw[connection[0]]
        coord2 = coords_hw[connection[1]]
        coords = np.stack([coord1, coord2])
        # 0.5, 0.5 is the center
        x = (coords[:, 0]-0.5)* W_scale+ W_offset
        y = -(coords[:, 1]-0.5)* H_scale+ H_offset
        
        #axis.plot( x , y , color=color, linewidth=linewidth)
        ret_coor.append( [x[1], y[1]] )
        if connection[1] == 1:
            ret_coor.append( [x[0], y[0]] )

    return np.asarray(ret_coor)

def draw(keypoint_coord3d_v, index, azim_z, elev_e):
    ret_coor = plot_hand(keypoint_coord3d_v, linewidth='3')
    return ret_coor

cpmf = open("fixed_view/cpm_label.txt",'w')
for i in range(len(dic))[0:]:
    keypoint_coord3d_v = dic[str(i).zfill(7)]['perspective']
    ret_coor = draw(keypoint_coord3d_v, i, 90, -90)
    cpm_line = [return_img_path(i)]
    xminymin = np.min( ret_coor , axis=0 )
    xmin, ymin = xminymin[0] , xminymin[1]
    xmaxymax = np.max( ret_coor , axis=0 )
    xmax, ymax = xmaxymax[0] , xmaxymax[1]
    #print(ret_coor.shape)
    #print(xmaxymax)
    #print(ymax)
    cpm_line.append("%d"%(int(ymin)-1))
    cpm_line.append("%d"%(int(xmin)-1))
    cpm_line.append("%d"%(int(ymax)+1))
    cpm_line.append("%d"%(int(xmax)+1))
    for i in range(21): # 21 keypoints
        cpm_line.append("%d"%(int(ret_coor[i,1])))
        cpm_line.append("%d"%(int(ret_coor[i,0])))
    cpm_line = " ".join(cpm_line)
    cpmf.write(cpm_line)
    cpmf.write("\n")


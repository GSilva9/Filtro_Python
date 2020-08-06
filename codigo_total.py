# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:27:00 2019

@author: giova
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#yy = cv2.imread('teste3.jpg')

imagem = cv2.imread('teste.jpg')

#plt.imshow(imagem, 'testando')


#cv2.imshow('teste', imagem)
px = np.zeros((36,45,3), dtype=int)
px = imagem

px = np.pad(imagem, ((3, 3),(3, 3),(0, 0)), 'edge')
 
print ("altura1 (height): %d pixels" % (imagem.shape[0]))
print ("largura1 (width): %d pixels" % (imagem.shape[1]))
#print ("Canais (channels): %d"      % (imagem.shape[2]))
    
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

'''
import timeit
total_time = timeit.timeit('[i for i in range(20000)]', number=20000);
print(total_time)
'''
l = 20000;
c = 1;

# Criar variavél e setando o valor inicial em zero
eq1 = np.zeros((l, c), dtype=int);
eq2 = np.zeros((l, c), dtype=int);
eq3 = np.zeros((l, c), dtype=int);
eq4 = np.zeros((l, c), dtype=int);
eq5 = np.zeros((l, c), dtype=int);
eq6 = np.zeros((l, c), dtype=int);
eq7 = np.zeros((l, c), dtype=int);
eq8 = np.zeros((l, c), dtype=int);
eq9 = np.zeros((l, c), dtype=int);
eq10 = np.zeros((l, c), dtype=int);
eq11 = np.zeros((l, c), dtype=int);
eq12 = np.zeros((l, c), dtype=int);
eq13 = np.zeros((l, c), dtype=int);
eq14 = np.zeros((l, c), dtype=int);
eq15 = np.zeros((l, c), dtype=int);
W = np.zeros((l, c), dtype=int);
A = np.zeros((l, c), dtype=int);
B = np.zeros((l, c), dtype=int);
D = np.zeros((l, c), dtype=int);
E = np.zeros((l, c), dtype=int);
F = np.zeros((l, c), dtype=int);
C = np.zeros((l, c), dtype=int);
G = np.zeros((l, c), dtype=int);

# Gera 6 dados randomicos para estimar 20000 pixel

altura1 = 36
largura1 = 38
profundidade = 3

p = np.zeros((altura1,largura1,profundidade), dtype=int)
p2 = np.zeros((altura1,largura1,profundidade), dtype=int)
p3 = np.zeros((altura1,largura1,profundidade), dtype=int)
p4 = np.zeros((altura1,largura1,profundidade), dtype=int)
p5 = np.zeros((altura1,largura1,profundidade), dtype=int)
p6 = np.zeros((altura1,largura1,profundidade), dtype=int)
p7 = np.zeros((altura1,largura1,profundidade), dtype=int)
p8 = np.zeros((altura1,largura1,profundidade), dtype=int)
p9 = np.zeros((altura1,largura1,profundidade), dtype=int)
p10 = np.zeros((altura1,largura1,profundidade), dtype=int)
p11 = np.zeros((altura1,largura1,profundidade), dtype=int)
p12 = np.zeros((altura1,largura1,profundidade), dtype=int)
p13 = np.zeros((altura1,largura1,profundidade), dtype=int)
p14 = np.zeros((altura1,largura1,profundidade), dtype=int)
p15 = np.zeros((altura1,largura1,profundidade), dtype=int)


altura = 36
largura = 39*16

p100 = np.zeros((altura,largura,profundidade), dtype=int)
p101 = np.zeros((altura,largura,profundidade), dtype=int)
p102 = np.zeros((altura,largura,profundidade), dtype=int)
p103 = np.zeros((altura,largura,profundidade), dtype=int)
p104 = np.zeros((altura,largura,profundidade), dtype=int)
p105 = np.zeros((altura,largura,profundidade), dtype=int)
p106 = np.zeros((altura,largura,profundidade), dtype=int)
p107 = np.zeros((altura,largura,profundidade), dtype=int)
p108 = np.zeros((altura,largura,profundidade), dtype=int)
p109 = np.zeros((altura,largura,profundidade), dtype=int)
p110 = np.zeros((altura,largura,profundidade), dtype=int)
p111 = np.zeros((altura,largura,profundidade), dtype=int)
p112 = np.zeros((altura,largura,profundidade), dtype=int)
p113 = np.zeros((altura,largura,profundidade), dtype=int)
p114 = np.zeros((altura,largura,profundidade), dtype=int)


phori = np.zeros(((36),(39*16),3), dtype=int)
phori2 = np.zeros(((42),(39*16),3), dtype=int)
pfinal = np.zeros(((36*16),(39*16),3), dtype=int)


#---- HORIZONTAL -------
for j in range (0, 3):
    for r in range(0, 36):
        for i in range (3, 35):
            C[i] = px[r][i-3][j]
            B[i] = px[r][i-2][j]
            A[i] = px[r][i-1][j]
            W[i] = px[r][i][j]
            D[i] = px[r][i+1][j]
            E[i] = px[r][i+2][j]
            F[i] = px[r][i+3][j]
            G[i] = px[r][i+4][j] 

        #------ EQUAÇÕES -------

            eq1 = ((0*C[i]) + (1*B[i]) + (-3*A[i]) + (63*W[i]) + (4*D[i]) + (-2*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq1)
            p[r][i-3][j] = eq1
            
        #--------
        
            eq2 = ((-1*C[i]) + (2*B[i]) + (-5*A[i]) + (62*W[i]) + (8*D[i]) + (-3*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq2)
            p2[r][i-3][j] = eq2
                                 
        #---------
            eq3 = ((-1*C[i]) + (3*B[i]) + (-8*A[i]) + (60*W[i]) + (13*D[i]) + (-4*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq3)
            p3[r][i-3][j] = eq3

        #---------
            eq4 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (58*W[i]) + (17*D[i]) + (-5*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq4)
            p4[r][i-3][j] = eq4

        #---------
            eq5 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (52*W[i]) + (26*D[i]) + (-8*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq5)
            p5[r][i-3][j] = eq5

        #---------
            eq6 = ((-1*C[i]) + (3*B[i]) + (-9*A[i]) + (47*W[i]) + (31*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq6)
            p6[r][i-3][j] = eq6

        #----------
            eq7 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (45*W[i]) + (34*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq7)

            p7[r][i-3][j] = eq7
            
        #---------
            eq8 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (40*W[i]) + (40*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq8)
            p8[r][i-3][j] = eq8

        #---------
            eq9 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (34*W[i]) + (45*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq9)
            p9[r][i-3][j] = eq9

        #---------
            eq10 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (31*W[i]) + (47*D[i]) + (-9*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq10)

            p10[r][i-3][j] = eq10

        #---------
            eq11 = ((-1*C[i]) + (3*B[i]) + (-8*A[i]) + (26*W[i]) + (52*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq11)
            p11[r][i-3][j] = eq11
        
        #---------
            eq12 = ((0*C[i]) + (1*B[i]) + (-5*A[i]) + (17*W[i]) + (58*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq12)
            
            p12[r][i-3][j] = eq12

        #---------
            eq13 = ((0*C[i]) + (1*B[i]) + (-4*A[i]) + (13*W[i]) + (60*D[i]) + (-8*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq13)
            p13[r][i-3][j] = eq13

        #---------
            eq14 = ((0*C[i]) + (1*B[i]) + (-3*A[i]) + (8*W[i]) + (62*D[i]) + (-5*E[i]) + (2*F[i]) + (-1*G[i]))/64
            #print(eq14)
            p14[r][i-3][j] = eq14

        #---------
            eq15 = ((0*C[i]) + (1*B[i]) + (-2*A[i]) + (4*W[i]) + (63*D[i]) + (-3*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq15)
            p15[r][i-3][j] = eq15


for j in range (0, 3):
    for r in range(0, 36):
        for i in range (0, 32):
            phori[r][i*16][j] = imagem[r][i+3][j]
            phori[r][i*16+1][j] = p[r][i+3][j]
            phori[r][i*16+2][j] = p2[r][i+3][j]
            phori[r][i*16+3][j] = p3[r][i+3][j]
            phori[r][i*16+4][j] = p4[r][i+3][j]
            phori[r][i*16+5][j] = p5[r][i+3][j]
            phori[r][i*16+6][j] = p6[r][i+3][j]
            phori[r][i*16+7][j] = p7[r][i+3][j]
            phori[r][i*16+8][j] = p8[r][i+3][j]
            phori[r][i*16+9][j] = p9[r][i+3][j]
            phori[r][i*16+10][j] = p10[r][i+3][j]
            phori[r][i*16+11][j] = p11[r][i+3][j]
            phori[r][i*16+12][j] = p12[r][i+3][j]
            phori[r][i*16+13][j] = p13[r][i+3][j]
            phori[r][i*16+14][j] = p14[r][i+3][j]
            phori[r][i*16+15][j] = p15[r][i+3][j]
            
phori2 = np.zeros((42,624,3), dtype=int)
phori2 = imagem

phori2 = np.pad(imagem, ((3, 3),(3, 3),(0, 0)), 'edge')


array = np.array(phori, dtype=np.uint8)     
im = Image.fromarray(array)

im.show()
print(phori)            

            
#------ Vertical
for j in range (0, 3):
    for i in range(0, 624):
        for r in range (3, 38):
            C[i] = phori2[r-3][i][j]
            B[i] = phori2[r-2][i][j]
            A[i] = phori2[r-1][i][j]
            W[i] = phori2[r][i][j]  
            D[i] = phori2[r+1][i][j]
            E[i] = phori2[r+2][i][j]
            F[i] = phori2[r+3][i][j]
            G[i] = phori2[r+4][i][j] 

        #------ EQUAÇÕES -------

            eq1 = ((0*C[i]) + (1*B[i]) + (-3*A[i]) + (63*W[i]) + (4*D[i]) + (-2*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq1)
            p100[r-3][i][j] = eq1            
        #--------
        
            eq2 = ((-1*C[i]) + (2*B[i]) + (-5*A[i]) + (62*W[i]) + (8*D[i]) + (-3*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq2)
            p101[r-3][i][j] = eq2
                                 
        #---------
            eq3 = ((-1*C[i]) + (3*B[i]) + (-8*A[i]) + (60*W[i]) + (13*D[i]) + (-4*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq3)
            p102[r-3][i][j] = eq3

        #---------
            eq4 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (58*W[i]) + (17*D[i]) + (-5*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq4)
            p103[r-3][i][j] = eq4

        #---------
            eq5 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (52*W[i]) + (26*D[i]) + (-8*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq5)
            p104[r-3][i][j] = eq5

        #---------
            eq6 = ((-1*C[i]) + (3*B[i]) + (-9*A[i]) + (47*W[i]) + (31*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq6)
            p105[r-3][i][j] = eq6

        #----------
            eq7 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (45*W[i]) + (34*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq7)
            p106[r-3][i][j] = eq7
            
        #---------
            eq8 = ((-1*C[i]) + (4*B[i]) + (-11*A[i]) + (40*W[i]) + (40*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq8)
            p107[r-3][i][j] = eq8

        #---------
            eq9 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (34*W[i]) + (45*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq9)
            p108[r-3][i][j] = eq9

        #---------
            eq10 = ((-1*C[i]) + (4*B[i]) + (-10*A[i]) + (31*W[i]) + (47*D[i]) + (-9*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq10)
            p109[r-3][i][j] = eq10

        #---------
            eq11 = ((-1*C[i]) + (3*B[i]) + (-8*A[i]) + (26*W[i]) + (52*D[i]) + (-11*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq11)
            p110[r-3][i][j] = eq11
        
        #---------
            eq12 = ((0*C[i]) + (1*B[i]) + (-5*A[i]) + (17*W[i]) + (58*D[i]) + (-10*E[i]) + (4*F[i]) + (-1*G[i]))/64
            #print(eq12)
            p111[r-3][i][j] = eq12

        #---------
            eq13 = ((0*C[i]) + (1*B[i]) + (-4*A[i]) + (13*W[i]) + (60*D[i]) + (-8*E[i]) + (3*F[i]) + (-1*G[i]))/64
            #print(eq13)
            p112[r-3][i][j] = eq13

        #---------
            eq14 = ((0*C[i]) + (1*B[i]) + (-3*A[i]) + (8*W[i]) + (62*D[i]) + (-5*E[i]) + (2*F[i]) + (-1*G[i]))/64
            #print(eq14)
            p113[r-3][i][j] = eq14

        #---------
            eq15 = ((0*C[i]) + (1*B[i]) + (-2*A[i]) + (4*W[i]) + (63*D[i]) + (-3*E[i]) + (1*F[i]) + (0*G[i]))/64
            #print(eq15)
            p114[r-3][i][j] = eq15

            
for j in range (0, 3):
    for r in range(0, 36):
        for i in range (0, 624):            
            pfinal[r*16][i][j] = phori2[r][i][j]
            pfinal[r*16+1][i][j] = p100[r][i][j]
            pfinal[r*16+2][i][j] = p101[r][i][j]
            pfinal[r*16+3][i][j] = p102[r][i][j]
            pfinal[r*16+4][i][j] = p103[r][i][j]
            pfinal[r*16+5][i][j] = p104[r][i][j]
            pfinal[r*16+6][i][j] = p105[r][i][j]
            pfinal[r*16+7][i][j] = p106[r][i][j]
            pfinal[r*16+8][i][j] = p107[r][i][j]
            pfinal[r*16+9][i][j] = p108[r][i][j]
            pfinal[r*16+10][i][j] = p109[r][i][j]
            pfinal[r*16+11][i][j] = p110[r][i][j]
            pfinal[r*16+12][i][j] = p111[r][i][j]
            pfinal[r*16+13][i][j] = p112[r][i][j]
            pfinal[r*16+14][i][j] = p113[r][i][j]
            pfinal[r*16+15][i][j] = p114[r][i][j]


array = np.array(pfinal, dtype=np.uint8)     
im = Image.fromarray(array)

#im.show()
im.save('filtro1final2.jpg')


'''
array = np.array(p, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro1.jpg')

array = np.array(p2, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro2.jpg')

array = np.array(p3, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro3.jpg')

array = np.array(p4, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro4.jpg')

array = np.array(p5, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro5.jpg')

array = np.array(p6, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro6.jpg')

array = np.array(p7, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro7.jpg')

array = np.array(p8, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro8.jpg')

array = np.array(p9, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro9.jpg')
            
array = np.array(p10, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro10.jpg')

array = np.array(p11, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro11.jpg')

array = np.array(p12, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro12.jpg')

array = np.array(p13, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro13.jpg')

array = np.array(p14, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro14.jpg')


array = np.array(p15, dtype=np.uint8)     
im = Image.fromarray(array)
im.save('filtro15.jpg')
 '''           
#plt.plot(W, c='g');
#plt.plot(eq1, c='r');
#plt.show();













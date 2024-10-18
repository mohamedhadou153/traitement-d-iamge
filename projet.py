import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from matplotlib import pyplot as plt
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
#from PIL import Image, ImageTk

def select_image():

    file_path = filedialog.askopenfilename()
    if file_path:
        global original_image, original_hist
        original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
        affichage(original_image)
    process_menu0.entryconfig(0, state="normal")
    process_menu0.entryconfig(5, state="normal")
    process_menu0.entryconfig(1, state="normal")
    process_menu0.entryconfig(2, state="normal")
    process_menu0.entryconfig(3, state="normal")
    process_menu0.entryconfig(4, state="normal")
    process_menu1.entryconfig(0, state="normal")
    process_menu1.entryconfig(1, state="normal")
    process_menu1.entryconfig(2, state="normal")
    process_menu1.entryconfig(3, state="normal")
    process_menu1.entryconfig(4, state="normal")
    process_menu2.entryconfig(0, state="normal")
    process_menu2.entryconfig(1, state="normal")
    process_menu2.entryconfig(2, state="normal")
    process_menu2.entryconfig(3, state="normal")
    process_menu2.entryconfig(4, state="normal")
    process_menu3.entryconfig(0, state="normal")
    process_menu3.entryconfig(1, state="normal")
    process_menu3.entryconfig(2, state="normal")
    process_menu3.entryconfig(3, state="normal")
    process_menu4.entryconfig(0, state="normal")
    process_menu4.entryconfig(1, state="normal")
    process_menu4.entryconfig(2, state="normal")
    process_menu4.entryconfig(3, state="normal")
    process_menu4.entryconfig(4, state="normal")
    process_menu4.entryconfig(5, state="normal")
    process_menu4.entryconfig(6, state="normal")
    process_menu4.entryconfig(7, state="normal")
    process_menu4.entryconfig(8, state="normal")
    process_menu5.entryconfig(0, state="normal")
    process_menu5.entryconfig(1, state="normal")
    process_menu6.entryconfig(0, state="normal")
def affichage(img):

    if hasattr(window, 'canvas'):
        window.canvas.get_tk_widget().destroy()

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off axis
    plt.tight_layout()

    # Convert Matplotlib figure to Tkinter canvas
    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas.draw()
    window.canvas = canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
def affichage_histogramme_hough(image, hist, imlab, histlab, lines, xy_long):
    # Convertir l'image OpenCV en format compatible avec Tkinter
    if hasattr(window, 'canvas'):
        window.canvas.get_tk_widget().destroy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Réduire les dimensions de la figure

    # Display original image and histogram in the first row
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 1].plot(original_hist, color='black')
    axs[0, 1].set_title("Histogramme original")
    axs[0, 1].set_xlim([0, 256])

    # Display the processed image and histogram in the second row
    axs[1, 0].imshow(image, cmap='gray')
    axs[1, 0].set_title(imlab)
    axs[1, 1].plot(hist, color='black')
    axs[1, 1].set_title(histlab)
    axs[1, 1].set_xlim([0, 256])

    for xy in lines[:, 0]:
        axs[1, 0].plot([xy[0], xy[2]], [xy[1], xy[3]], linewidth=2, color='green')
        axs[1, 0].plot(xy[0], xy[1], 'x', linewidth=2, color='yellow')
        axs[1, 0].plot(xy[2], xy[3], 'x', linewidth=2, color='red')

    axs[1, 0].plot([xy_long[0][0], xy_long[1][0]], [xy_long[0][1], xy_long[1][1]], linewidth=2, color='red')


    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    window.canvas = canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,expand=1)
def affichage_image_histogramme(img, hist, imlab, histlab):

    if hasattr(window, 'canvas'):
        window.canvas.get_tk_widget().destroy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Réduire les dimensions de la figure

    # Display original image and histogram in first row
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 1].plot(original_hist, color='black')
    axs[0, 1].set_title("Histogramme original")
    axs[0, 1].set_xlim([0, 256])

    # Display processed image and histogram in second row
    axs[1, 0].imshow(img, cmap='gray')
    axs[1, 0].set_title(imlab)
    axs[1, 1].plot(hist, color='black')
    axs[1, 1].set_title(histlab)
    axs[1, 1].set_xlim([0, 256])

    plt.tight_layout()

    # Convert Matplotlib figure to Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    window.canvas = canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
def affichage_image_histogramme_harris(img, hist, imlab, histlab, x, y):
    if hasattr(window, 'canvas'):
        window.canvas.get_tk_widget().destroy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Réduire les dimensions de la figure

    # Display original image and histogram in the first row
    axs[0, 0].imshow(original_image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 1].plot(original_hist, color='black')
    axs[0, 1].set_title("Histogramme original")
    axs[0, 1].set_xlim([0, 256])

    # Display the processed image and histogram in the second row
    axs[1, 0].imshow(img, cmap='gray')
    axs[1, 0].set_title(imlab)
    axs[1, 1].plot(hist, color='black')
    axs[1, 1].set_title(histlab)
    axs[1, 1].set_xlim([0, 256])

    # Overlay points on the processed image
    axs[1, 0].scatter(y, x, c='r', marker='.')
    axs[1, 0].set_xlim([0, img.shape[1]])  # Adjust xlim based on image width
    axs[1, 0].set_ylim([img.shape[0], 0])  # Adjust ylim based on image height

    plt.tight_layout()

    # Convert Matplotlib figure to Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    window.canvas = canvas
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
def egalisation_histogramme(img):
    equalized_img = cv2.equalizeHist(img)
    hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
    return equalized_img, hist_equalized
def appliquer_filtre_median(image):
    image = image.astype(np.float64)
    n, m = image.shape
    img = image.copy()

    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            v = fenetre.flatten()
            v.sort()
            a = np.median(v)
            img[i, j] = a

    img = img.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_pyramidale(image):
    image = image.astype(np.float64)
    n, m = image.shape
    img = np.zeros_like(image)

    H = (1/81) * np.array([[1, 2, 3, 2, 1],
                          [2, 4, 6, 4, 2],
                          [3, 6, 9, 6, 3],
                          [2, 4, 6, 4, 2],
                          [1, 2, 3, 2, 1]])

    for x in range(2, n - 2):
        for y in range(2, m - 2):
            fenetre = image[x - 2:x + 3, y - 2:y + 3]
            v = fenetre * H
            img[x, y] = np.sum(v)

    img = np.uint8(img)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_sobel(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)
    img = gradient_magnitude.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_prewitt(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)
    img = gradient_magnitude.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_roberts(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)
    img = gradient_magnitude.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_laplacien(image):
    image = image.astype(np.float64)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    output = np.abs(laplacian)
    img = output.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_gradient(image):
    image = image.astype(np.float64)
    output_hor = cv2.filter2D(image, cv2.CV_64F, np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]))
    output_ver = cv2.filter2D(image, cv2.CV_64F, np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]))
    output = np.sqrt(output_hor*2 + output_ver*2)
    img = output.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def apply_filter_Fphb(image):
    # Appliquer la transformation de Fourier
    F = np.fft.fftshift(np.fft.fft2(image))
    # Calculer la taille de l'image
    M, N = F.shape
    # Initialiser le filtre passe-haut
    H1 = np.ones((M, N))
    D0 = 1
    M2 = round(M/2)
    N2 = round(N/2)
    H1[M2-D0:M2+D0, N2-D0:N2+D0] = 0
    # Initialiser la variable G
    G = np.zeros_like(F, dtype=complex)
    # Paramètres du filtre Butterworth
    n = 3
    # Appliquer le filtre
    for i in range(M):
        for j in range(N):
            H = 1 / (1 + (H1[i, j] / D0) * (2 * n))
            G[i, j] = F[i, j] * H
    # Effectuer la transformée inverse
    g = np.fft.ifft2(np.fft.ifftshift(G))
    # Afficher l'image filtrée
    result = 255 - np.abs(g)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def apply_filter_Fph(image):
    # Appliquer la transformation de Fourier
    F = np.fft.fftshift(np.fft.fft2(image))
    # Calculer la taille de l'image
    M, N = F.shape
    # Initialiser la variable G
    G = np.zeros_like(F, dtype=complex)
    # Initialiser le filtre passe-haut
    H1 = np.ones((M, N))
    D0 = 1
    M2 = round(M/2)
    N2 = round(N/2)
    H1[M2-D0:M2+D0, N2-D0:N2+D0] = 0
    # Appliquer le filtre
    for i in range(M):
        for j in range(N):
            G[i, j] = F[i, j] * H1[i, j]
    # Effectuer la transformée inverse
    g = np.fft.ifft2(np.fft.ifftshift(G))
    # Afficher l'image filtrée
    result = 255 - np.abs(g)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def apply_filter_Fpbb(image):
    # Appliquer la transformation de Fourier
    F = np.fft.fftshift(np.fft.fft2(image))
    # Calculer la taille de l'image
    M, N = F.shape
    # Initialiser le filtre passe-bas
    H0 = np.zeros((M, N))
    D0 = 1
    M2 = round(M/2)
    N2 = round(N/2)
    H0[M2-D0:M2+D0, N2-D0:N2+D0] = 1
    # Initialiser la variable G
    G = np.zeros_like(F, dtype=complex)
    # Appliquer le filtre
    for i in range(M):
        for j in range(N):
            G[i, j] = F[i, j] * H0[i, j]
    # Effectuer la transformée inverse
    g = np.fft.ifft2(np.fft.ifftshift(G))
    # Afficher l'image filtrée
    result = np.abs(g)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def apply_filter_Fpb(image):
    # Appliquer la transformation de Fourier
    F = np.fft.fftshift(np.fft.fft2(image))
    # Calculer la taille de l'image
    M, N = F.shape
    # Initialiser le filtre passe-bas
    H0 = np.zeros((M, N))
    D0 = 1
    M2 = round(M/2)
    N2 = round(N/2)
    H0[M2-D0:M2+D0, N2-D0:N2+D0] = 1
    # Initialiser la variable G
    G = np.zeros_like(F, dtype=complex)
    # Appliquer le filtre
    for i in range(M):
        for j in range(N):
            G[i, j] = F[i, j] * H0[i, j]
    # Effectuer la transformée inverse
    g = np.fft.ifft2(np.fft.ifftshift(G))
    # Afficher l'image filtrée
    result = np.abs(g)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_moyenneur(image):
    taille_noyau=10
    image = image.astype(np.float64)
    n, m = image.shape
    b = np.zeros_like(image)

    for x in range(taille_noyau // 2, n - taille_noyau // 2):
        for y in range(taille_noyau // 2, m - taille_noyau // 2):
            fenetre = image[x - taille_noyau // 2:x + taille_noyau // 2 + 1, y - taille_noyau // 2:y + taille_noyau // 2 + 1]
            v = fenetre.flatten()
            moyenne = np.mean(v)
            b[x, y] = moyenne
    img = b.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_gaussien(image):
    taille_noyau=3
    image = image.astype(np.float64)
    n, m = image.shape
    b = np.zeros_like(image)


    sigma = 1  # Ajustez la valeur de sigma selon vos besoins

    H = np.zeros((taille_noyau, taille_noyau))
    for i in range(taille_noyau):
        for j in range(taille_noyau):
            H[i, j] = np.exp(-((i - taille_noyau // 2)**2 + (j - taille_noyau // 2)**2) / (2 * sigma**2))

    H /= np.sum(H)

    for x in range(taille_noyau // 2, n - taille_noyau // 2):
        for y in range(taille_noyau // 2, m - taille_noyau // 2):
            fenetre = image[x - taille_noyau // 2:x + taille_noyau // 2 + 1, y - taille_noyau // 2:y + taille_noyau // 2 + 1]
            v = fenetre.flatten()
            convolution = np.sum(v * H.flatten())
            b[x, y] = convolution
    img = b.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def appliquer_filtre_conique(image):
    image = image.astype(np.float64)
    n, m = image.shape
    b = np.zeros_like(image)

    H = (1/25) * np.array([[0, 0, 1, 0, 0],
                           [0, 2, 2, 2, 0],
                           [1, 2, 5, 2, 1],
                           [0, 2, 2, 2, 0],
                           [0, 0, 1, 0, 0]])

    for x in range(2, n - 2):
        for y in range(2, m - 2):
            fenetre = image[x - 2:x + 3, y - 2:y + 3]
            v = fenetre * H
            b[x, y] = np.sum(v)
    img = np.uint8(b)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def harris(image):
    lambda_val = 0.04
    sigma = 1
    threshold = 200
    r = 6
    w = 5 * sigma

    # Assuming 'image' is your image data
    m, n = image.shape
    imd = image.astype(float)

    # Derivative filters
    dx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

    # Horizontal derivative (Sobel filter)
    dy = dx.T  # Vertical derivative (transposed Sobel filter)

    # Gaussian filter
    g = gaussian_filter(np.ones((max(1, int(np.fix(w))), max(1, int(np.fix(w))))), sigma)
    
    # Image gradients
    Ix = convolve(imd, dx, mode='constant', cval=0.0)
    Iy = convolve(imd, dy, mode='constant', cval=0.0)
    
    # Squared gradients convolved with Gaussian
    Ix2 = convolve(Ix**2, g, mode='constant', cval=0.0)
    Iy2 = convolve(Iy**2, g, mode='constant', cval=0.0)
    Ixy = convolve(Ix * Iy, g, mode='constant', cval=0.0)
    # Calculate Harris corner response
    det_M = Ix2 * Iy2 - Ixy**2
    tr_M = Ix2 + Iy2
    R = det_M - lambda_val * tr_M**2
    R1 = (1000 / (1 + np.max(R))) * R
    u, v = np.where(R1 <= threshold)
    nb = len(u)
    
    for k in range(nb):
        R1[u[k], v[k]] = 0
    
    R11 = np.zeros((m + 2 * r, n + 2 * r))
    R11[r:r + m, r:r + n] = R1
    
    m1, n1 = R11.shape
    
    for i in range(r + 1, m1 - r):
        for j in range(r + 1, n1 - r):
            window = R11[i - r:i + r, j - r:j + r]
            ma = np.max(window)
            if window[r, r] < ma:
                R11[i, j] = 0
    img = imd.astype(np.uint8)
    R11 = R11[r+1:m+r, r+1:n+r]
    x, y = np.where(R11)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized,x,y
def susan(image):
    #image = np.double(im)
    n, m = image.shape
    rayon = 2
    alpha = 50
    r = 2
    alpha = alpha / 100

    mask = np.zeros((2 * rayon + 1, 2 * rayon + 1))
    b = np.ones((rayon + 1, rayon + 1))

    for i in range(1, rayon + 2):
        for j in range(1, rayon + 2):
            if rayon == 1:
                if j > i:
                    b[i - 1, j - 1] = 0
            else:
                if j > i + 1:
                    b[i - 1, j - 1] = 0

    mask[0:rayon + 1, rayon:2 * rayon + 1] = b
    mask[0:rayon + 1, 0:rayon + 1] = np.rot90(b)
    mask0 = np.copy(mask)
    mask0 = np.flipud(mask0)
    mask = mask0 + mask
    mask[rayon, :] = mask[rayon, :] - 1
    max_reponse = np.sum(mask)

    f = np.zeros((n, m))

    for i in range(rayon + 1, n - rayon):
        for j in range(rayon + 1, m - rayon):
            image_courant = image[i - rayon:i + rayon + 1, j - rayon:j + rayon + 1]
            image_courant_mask = image_courant * mask

            intensite_centrale = image_courant_mask[rayon, rayon]
            s = np.exp(-1 * (((image_courant_mask - intensite_centrale) / max_reponse) ** 6))
            somme = np.sum(s)

            if intensite_centrale == 0:
                somme = somme - np.sum(mask == 0)

            f[i, j] = somme

    ff = f[rayon + 1:n - (rayon + 1), rayon + 1:m - (rayon + 1)]
    minf = np.min(ff)
    maxf = np.max(f)
    fff = np.copy(f)

    d = 2 * r + 1
    temp1 = round(n / d)
    if (temp1 - n / d) < 0.5 and (temp1 - n / d) > 0:
        temp1 = temp1 - 1
    temp2 = round(m / d)
    if (temp2 - m / d) < 0.5 and (temp2 - m / d) > 0:
        temp2 = temp2 - 1

    fff[n:temp1 * d + d, m:temp2 * d + d] = 0

    for i in range(r + 1, d, temp1 * d + d):
        for j in range(r + 1, d, temp2 * d + d):
            window = fff[i - r:i + r + 1, j - r:j + r + 1]
            window0 = np.copy(window)
            xx, yy = np.where(window0 == 0)

            for k in range(len(xx)):
                window0[xx[k], yy[k]] = np.max(np.max(window0))

            minwindow = np.min(np.min(window0))
            y, x = np.where((minwindow != window) & (window <= minf + alpha * (maxf - minf)) & (window > 0))
            u, v = np.where(minwindow == window)


            if len(u) > 1:
                for l in range(1, len(u)):
                    fff[i - r - 1 + u[l], j - r - 1 + v[l]] = 0

            if len(x) != 0:
                for l in range(len(y)):
                    fff[i - r - 1 + y[l], j - r - 1 + x[l]] = 0

    seuil = minf + alpha * (maxf - minf)
    u,v = np.where((minf <= fff) & (fff <= seuil))
    #u = indices[0]
    #v = indices[1]


    hist_equalized = cv2.calcHist([image], [0], None, [256], [0, 256])
    return image, hist_equalized,u,v
def chapeau_haut_b(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Opening operation
    O = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

    # Compute the top hat transform
    image_tophat = np.subtract(image.astype(float), O.astype(float))
    img = np.uint8(image_tophat)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def chapeau_haut_n(image):

    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Closing operation
    F = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    # Compute the top hat transform
    image_tophat = np.subtract(F.astype(float), image.astype(float))
    img = np.uint8(image_tophat)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def gradient_morph(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Erode and dilate the image
    erodedI = cv2.erode(image, se)
    dilatedI = cv2.dilate(image, se)

    # Compute the morphological gradient
    image_gradient = np.subtract(dilatedI.astype(float), erodedI.astype(float))
    img = np.uint8(image_gradient)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def contour_externe(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Dilate the image
    dilatedI = cv2.dilate(image, se)

    # Compute the contour
    image_contour = np.subtract(dilatedI.astype(float), image.astype(float))
    img = np.uint8(image_contour)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def contour_interne(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Erode the image
    erodedI = cv2.erode(image, se)

    # Compute the contour
    image_contour = np.subtract(image.astype(float), erodedI.astype(float))
    img = np.uint8(image_contour)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def fermeture(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Closing operation
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def ouverture(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Opening operation
    img = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def erosion(image):
    # Structuring element of type line with length 11 pixels and angle 90 degrees
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))  # equivalent to strel('line', 11, 90) in MATLAB

    # Erode the image
    img = cv2.erode(image, se)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def dilatation(image):
    # Structuring element of type disk with radius = 4 pixels
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # equivalent to strel('disk', 4) in MATLAB

    # Erode the image
    img = cv2.erode(image, se)

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def NivGris(image):
    # Vérifier la dimension de l'image
    d = len(image.shape)
    
    # Convertir en niveaux de gris si l'image est en couleur
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if d == 3 else image

    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def Binaris(image):
    # Convertir l'image en niveau de gris si elle est en couleur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Appliquer la méthode d'Otsu pour trouver le seuil optimal
    _, img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def luminosite(image, delta=50):
    # Assurez-vous que l'image est de type float
    image = image.astype(float)

    # Ajouter le delta à tous les pixels
    result = image + delta

    # Limiter les valeurs entre 0 et 255
    result[result > 255] = 255
    result[result < 0] = 0

    # Convertir l'image en type uint8
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def contraste(image, a=5, b=128):
    # Assurez-vous que l'image est de type float
    image = image.astype(float)

    # Appliquer la transformation de contraste
    result = (image - b) * a + b

    # Limiter les valeurs entre 0 et 255
    result[result > 255] = 255
    result[result < 0] = 0

    # Convertir l'image en type uint8
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def negatif(image):
    # Assurez-vous que l'image est de type float
    image = image.astype(float)

    # Appliquer la transformation négative
    result = -image + 255

    # Limiter les valeurs entre 0 et 255
    result[result > 255] = 255
    result[result < 0] = 0

    # Convertir l'image en type uint8
    img = result.astype(np.uint8)
    hist_equalized = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img, hist_equalized
def detec_droit(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=10, maxLineGap=5)
    
    # Find the longest line segment
    max_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        len_line = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if len_line > max_len:
            max_len = len_line
            xy_long = [(x1, y1), (x2, y2)]

    hist_equalized = cv2.calcHist([image], [0], None, [256], [0, 256])

    return image, hist_equalized, lines, xy_long
def traitement_affichage(choice):

    global original_image, original_hist
    if choice == 'Histogramme égalisé':
        processed_img, processed_hist = egalisation_histogramme(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après l'égalisation", 'Histogramme égalisé')
    elif choice == 'Appliquer le filtre médian':
        processed_img, processed_hist = appliquer_filtre_median(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après filtrage par filtre median", 'Histogramme de l image aprés application du filtre median')
    elif choice == 'Appliquer le filtre pyramidale':
        processed_img, processed_hist = appliquer_filtre_pyramidale(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après filtrage par Pyramidale", 'Histogramme de l image aprés application du filtre pyramidale')
    elif choice == 'Appliquer le filtre conique':
        processed_img, processed_hist = appliquer_filtre_conique(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après filtrage par conique", 'Histogramme de l image aprés application du filtre conique')
    elif choice == 'Niveau de gris':
        processed_img, processed_hist = NivGris(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image au niveau de gris", 'Histogramme de l image au niveau de gris')
    elif choice == 'Binairisation':
        processed_img, processed_hist = Binaris(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image binaire", 'Histogramme de l image binaire')
    elif choice == 'Contraste':
        processed_img, processed_hist = contraste(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Contraste d une image ", 'Histogramme de l image après contraste')
    elif choice == 'Luminosité':
        processed_img, processed_hist = luminosite(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Luminosité d une image", 'Histogramme de l image lumineuse')
    elif choice == 'Négatif':
        processed_img, processed_hist = negatif(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Négatif de l image", 'Histogramme du négatif de l image')
    elif choice == 'Gradient morphologique':
        processed_img, processed_hist = gradient_morph(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Gradient morphologique de l image", 'Histogramme du Gradient morphologique de l image')
    elif choice == 'Dilatation':
        processed_img, processed_hist = dilatation(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image dilatée", 'Histogramme de l image dilatée')
    elif choice == 'Erosion':
        processed_img, processed_hist = erosion(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image érodée", 'Histogramme de l image erodée')
    elif choice == 'Ouverture':
        processed_img, processed_hist = ouverture(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après l'ouverture", 'Histogramme de l image après l ouverture ')
    elif choice == 'Fermeture':
        processed_img, processed_hist = fermeture(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après la fermeture", 'Histogramme de l image après la fermeture ')
    elif choice == 'Contour interne':
        processed_img, processed_hist = contour_interne(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après un contour interne", 'Histogramme de l image après application d un contour interne')
    elif choice == 'Contour externe':
        processed_img, processed_hist = contour_externe(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après un contour externe", 'Histogramme de l image après application d un contour externe')
    elif choice == 'Chapeau_haut_n':
        processed_img, processed_hist = chapeau_haut_n(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après chapeau haut_n", 'Histogramme de l image après chapeau haut_n')
    elif choice == 'Chapeau_haut_b':
        processed_img, processed_hist = chapeau_haut_b(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après chapeau haut_b", 'Histogramme de l image après chapeau haut_b')
    elif choice == 'Harris':
        processed_img, processed_hist,x,y = harris(original_image)
        affichage_image_histogramme_harris(processed_img, processed_hist, "Image avec detection de contours par harris", 'Histogramme sans changement',x,y)
    elif choice == 'Susan':
        processed_img, processed_hist,x,y = susan(original_image)
        affichage_image_histogramme_harris(processed_img, processed_hist, "Image avec detection de contours par susan", 'Histogramme sans changement',x,y)
    elif choice == 'Appliquer le filtre Sobel':
        processed_img, processed_hist = appliquer_filtre_sobel(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image filtrée par Sobel", 'Histogramme de l image aprés application du filtre Sobel')
    elif choice == 'Appliquer le filtre Prewitt':
        processed_img, processed_hist = appliquer_filtre_prewitt(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image filtrée par Prewitt", 'Histogramme de l image aprés application du filtre Prewitt')
    elif choice == 'Appliquer le filtre Roberts':
        processed_img, processed_hist = appliquer_filtre_roberts(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image filtrée par Roberts", 'Histogramme de l image aprés application du filtre Roberts')
    elif choice == 'Appliquer le filtre Laplacien':
        processed_img, processed_hist = appliquer_filtre_laplacien(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image filtrée par Laplacien", 'Histogramme de l image aprés application du filtre Laplacien')
    elif choice == 'Appliquer le filtre Gradient':
        processed_img, processed_hist = appliquer_filtre_gradient(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image filtrée par Gradient", 'Histogramme de l image aprés application du gradient')
    elif choice == 'Appliquer fpbh':
        processed_img, processed_hist = apply_filter_Fphb(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après application du filtre fréquentiel FPBH", 'Histogramme de l image aprés application du filtre fréquentiel FPBH')
    elif choice == 'Appliquer fph':
        processed_img, processed_hist = apply_filter_Fph(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après application du filtre fréquentiel FPH", 'Histogramme de l image aprés application du filtre fréquentiel FPH')
    elif choice == 'Appliquer fpbb':
        processed_img, processed_hist = apply_filter_Fpbb(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après application du filtre fréquentiel FPBB", 'Histogramme de l image aprés application du filtre fréquentiel FPBB')
    elif choice == 'Appliquer fpb':
        processed_img, processed_hist = apply_filter_Fpbb(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après application du filtre fréquentiel par FPB", 'Histogramme de l image aprés application du filtre fréquentiel FPB')
    elif choice == 'Appliquer le filtre moyenneur':
        processed_img, processed_hist = appliquer_filtre_moyenneur(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après filtrage Moyenneur", 'Histogramme de l image aprés application du filtre moyenneur')
    elif choice == 'Appliquer le filtre gaussien':
        processed_img, processed_hist = appliquer_filtre_gaussien(original_image)
        affichage_image_histogramme(processed_img, processed_hist, "Image après filtrage Gaussien", 'Histogramme de l image aprés application du filtre gaussien')
    elif choice == 'Detection de droite':
        processed_img, processed_hist,lines,xy_long = detec_droit(original_image)
        affichage_histogramme_hough(processed_img, processed_hist, "Image avec detection de droite", 'Histogramme sans changement',lines,xy_long)

    else:
        messagebox.showerror("Erreur", "Choix invalide")



# Créer la fenêtre principale
window = tk.Tk()
window.title("Traitement d'images")
window.geometry("800x600")  # Taille de la fenêtre
ttk.Style().theme_use('clam')
# Créer un bouton pour choisir l'image
select_button = tk.Button(window, text="Choisir l'image", command=select_image)
select_button.pack(side=tk.TOP, pady=10)

# Créer un menu
menu = tk.Menu(window)
window.config(menu=menu)

# Menu pour le traitement du contraste et de la luminosité
process_menu1 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Filtres passe-bas", menu=process_menu1)
process_menu1.add_command(label="Appliquer le filtre médian", command=lambda: traitement_affichage("Appliquer le filtre médian"), state="disabled")
process_menu1.add_command(label="Appliquer le filtre moyenneur", command=lambda: traitement_affichage("Appliquer le filtre moyenneur"), state="disabled")
process_menu1.add_command(label="Appliquer le filtre gaussien", command=lambda: traitement_affichage("Appliquer le filtre gaussien"), state="disabled")
process_menu1.add_command(label="Appliquer le filtre conique", command=lambda: traitement_affichage("Appliquer le filtre conique"), state="disabled")
process_menu1.add_command(label="Appliquer le filtre pyramidale", command=lambda: traitement_affichage("Appliquer le filtre pyramidale"), state="disabled")

process_menu2 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Filtres passe-haut", menu=process_menu2)
process_menu2.add_command(label="Appliquer le filtre Laplacien", command=lambda: traitement_affichage("Appliquer le filtre Laplacien"), state="disabled")
process_menu2.add_command(label="Appliquer le filtre Sobel", command=lambda: traitement_affichage("Appliquer le filtre Sobel"), state="disabled")
process_menu2.add_command(label="Appliquer le filtre Prewitt", command=lambda: traitement_affichage("Appliquer le filtre Prewitt"), state="disabled")
process_menu2.add_command(label="Appliquer le filtre Roberts", command=lambda: traitement_affichage("Appliquer le filtre Roberts"), state="disabled")
process_menu2.add_command(label="Appliquer le filtre Gradient", command=lambda: traitement_affichage("Appliquer le filtre Gradient"), state="disabled")
process_menu3 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Filtres fréquentiels", menu=process_menu3)
process_menu3.add_command(label="Appliquer fpbh", command=lambda: traitement_affichage("Appliquer fpbh"), state="disabled")
process_menu3.add_command(label="Appliquer fph", command=lambda: traitement_affichage("Appliquer fph"), state="disabled")
process_menu3.add_command(label="Appliquer fpbb", command=lambda: traitement_affichage("Appliquer fpbb"), state="disabled")
process_menu3.add_command(label="Appliquer fpb", command=lambda: traitement_affichage("Appliquer fpb"), state="disabled")
process_menu0 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Transformations", menu=process_menu0)
process_menu0.add_command(label="Histogramme égalisé", command=lambda: traitement_affichage("Histogramme égalisé"), state="disabled")
process_menu0.add_command(label="Niveau de gris", command=lambda: traitement_affichage("Niveau de gris"), state="disabled")
process_menu0.add_command(label="Binairisation", command=lambda: traitement_affichage("Binairisation"), state="disabled")
process_menu0.add_command(label="Contraste", command=lambda: traitement_affichage("Contraste"), state="disabled")
process_menu0.add_command(label="Luminosité", command=lambda: traitement_affichage("Luminosité"), state="disabled")
process_menu0.add_command(label="Négatif", command=lambda: traitement_affichage("Négatif"), state="disabled")
process_menu4 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Morphologie", menu=process_menu4)
process_menu4.add_command(label="Dilatation", command=lambda: traitement_affichage("Dilatation"), state="disabled")
process_menu4.add_command(label="Erosion", command=lambda: traitement_affichage("Erosion"), state="disabled")
process_menu4.add_command(label="Ouverture", command=lambda: traitement_affichage("Ouverture"), state="disabled")
process_menu4.add_command(label="Fermeture", command=lambda: traitement_affichage("Fermeture"), state="disabled")
process_menu4.add_command(label="Contour interne", command=lambda: traitement_affichage("Contour interne"), state="disabled")
process_menu4.add_command(label="Contour externe", command=lambda: traitement_affichage("Contour externe"), state="disabled")
process_menu4.add_command(label="Gradient morphologique", command=lambda: traitement_affichage("Gradient morphologique"), state="disabled")
process_menu4.add_command(label="Chapeau_haut_n", command=lambda: traitement_affichage("Chapeau_haut_n"), state="disabled")
process_menu4.add_command(label="Chapeau_haut_b", command=lambda: traitement_affichage("Chapeau_haut_b"), state="disabled")
process_menu5 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Detection de contours", menu=process_menu5)
process_menu5.add_command(label="Harris", command=lambda: traitement_affichage("Harris"), state="disabled")
process_menu5.add_command(label="Susan", command=lambda: traitement_affichage("Susan"), state="disabled")
process_menu6 = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Hough transform", menu=process_menu6)
process_menu6.add_command(label="Detection de droite", command=lambda: traitement_affichage("Detection de droite"), state="disabled")

# Exécuter la boucle des événements Tkinter
window.mainloop()


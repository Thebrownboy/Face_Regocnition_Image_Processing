from commonfunctions import *



def median_filter(img , filter_size):
  for x in range(0 ,img.shape[0]-2):
    for y in range (0 , img.shape[1]-2):
      intensities = np.zeros((filter_size//2,filter_size//2))
      intensities= np.copy(img[x:x+3, y:y+3])
      img[x+1][y+1] = np.median(intensities)
  return img


def get_edges(img, thickness, blurring):
    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blurr the image in a way to smooth unnecessary edges
    gray_blur = cv2.medianBlur(gray, blurring)

    # Get the edges of the blurred img
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thickness, blurring)

    return edges



def save_user(name,img):
  path = './data'+name
  isExist = os.path.exists(path)

  if not isExist:
    os.makedirs(path)
    os.chdir(path)
    cv2.imwrite(name,img)

  if isExist:
    print("Name already exists")
  




# This may be done with segmentation in colors to select only 5-10 colors to apply on the image
def clear_colors(img):
    # We need to find a way to smooth the image and denoise it without hiding so many details. Use https://www.youtube.com/watch?v=CQPZhAVHsXg
    color = cv2.bilateralFilter(img , 9 , 300 , 300)
    return color


def cartoonization (img):    
    edges = get_edges(img , 11 ,5)
    color = clear_colors(img)
    cartoon = cv2.bitwise_and(color , color , mask=edges)
    return cartoon

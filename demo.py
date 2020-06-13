import cv2
from torch import load
from torchvision import transforms
from PIL import Image
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)

def draw_rectangle(img, points):
    x_min, x_max, y_min, y_max = points
    height, width, _ = img.shape
    # drawing rectangle
    start = (y_min, x_min)
    end = (y_max, x_max)
    color = (255, 255, 255)
    thickness = 3
    img = cv2.rectangle(img, start, end, color, thickness)
    # blurring outside parts
    mask = [img[0:x_min, :], img[x_max:width, :], img[x_min:x_max, 0:y_min], img[x_min:x_max, y_max:]]
    # designing outside blur for good looking effect lol
    for i in range(len(mask)):
        mask[i] = cv2.GaussianBlur(mask[i], (51, 51), 0)
    img[0:x_min, :] = mask[0]
    img[x_max:width, :] = mask[1]
    img[x_min:x_max, 0:y_min] = mask[2]
    img[x_min:x_max, y_max:] = mask[3]
    return img

# load model and define image transformation
model = load('model/3_class/model_v2_best.pth')
model.to('cpu')
model.eval()
transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5,0.5,0.5])
])


def run(model, img):
    # convert img color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get PIL instance from array
    img = Image.fromarray(img)
    # transform image
    img = transform(img)
    # reshape image
    img = img.reshape([1, 3, 224, 224])
    # get results
    result = [float(i) for i in model(img)[0]]
    # get most likely result
    index = result.index(max(result))
    # get probability
    denom = sum(np.exp(i) for i in result)
    prob = np.exp(result[index]) / denom

    return index, prob

while True:
    # get frame
    ret, img = vid.read()
    height, width, _ = img.shape
    # reshapping images
    x_min = int(100/480 * height)
    x_max = int(380/480 * height)
    y_min = int(96/640 * width)
    y_max = int(544/640*width)
    analyze_img = img[x_min:x_max, y_min:y_max, :]
    # get probability and class prediction
    index, prob = run(model, analyze_img)
    if index == 0:
        text = 'False. P = ' + str(prob)[:5]
        color = (0, 0, 255)
    else:
        text = ['_' ,'back', 'front'][index] + '. P = ' + str(prob)[:5]
        color = (255, 0, 0)
    points = [x_min, x_max, y_min, y_max]
    # draw rectangle
    img = draw_rectangle(img, points)
    # print label
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    org = (50,50)
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # showing image
    cv2.imshow('frame', img)

    # set quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
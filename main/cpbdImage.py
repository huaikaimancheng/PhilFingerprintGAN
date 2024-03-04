import cpbd
from scipy import ndimage
import os
def processCPDB(imagePath,filePath):
    images=os.listdir(imagePath)
    with open(filePath,'w') as file:
        for image in images:
            image_file=imagePath+"/"+image
            result_image = ndimage.imread(image_file,mode="L")
            score=cpbd.compute(result_image)
            file.write('{}\n'.format(score))


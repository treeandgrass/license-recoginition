from PIL import Image
import numpy as np
import os
from scipy.misc import imsave
from pandas import read_excel

# 图片工具类
class ImageUtils:
    def __init__(self):
        self.directory = []
    
    #图片截取    
    def narrow(self, image, ratio = 8):
        try:
            w = image.size[0]
            h = image.size[1]
            cutH = h // ratio
            cropImage = image.crop((0, cutH + cutH // 2, w, h))
            return cropImage
        except Exception as e:
            pass
    #灰度化    
    def L(self, image):
        try:
            im = image.convert('L')
            return im
        except Exception as e:
            pass
        
    #直方图    
    def histeq(self, im, nbr_bins = 256):
        imhist, bins = np.histogram(im.flatten(), nbr_bins, normed = True)
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        im2 = np.interp(im.flatten(), bins[:-1], cdf)
        return im2.reshape(im.shape).astype('int'), cdf
    # 均一化
    def imagenorm(self, image):
        _min = image.min()
        _max = image.max()
        newImage = (image - _min) / (_max - _min)
        return newImage
    
    #读取一张图片，将其转化为numpy对象
    def readImage (self, path):
        return np.array(Image.open(path))
    
    def getImageFiles(self, folder):
        self.directory = []
        self.getFileDir(folder)
        return self.directory
    
    #读取所有图片路径
    def getFileDir(self, folder):
        if os.path.exists(folder):
            if os.path.isdir(folder):
                dirs = os.listdir(folder)
                for d in dirs:
                    path = os.path.join(folder, d)
                    if os.path.isdir(path):
                        self.getFileDir(path)
                    else:
                        self.directory.append(path)
            else:
                self.directory.append(folder)
        else:
            print('{} not exists'.format(folder))
    def resize(self, path, shape = (150, 150)):
        return Image.open(path).resize(shape)
    #保存图片
    def saveImage(self, image, targetPath, _format = None):
        imsave(targetPath, image)
    
    #创建路径
    def createFolder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    # 综合处理图片
    def processImage(self, sourceFolder, targetFolder):
        self.createFolder(targetFolder)
        files = self.getImageFiles(sourceFolder)
        for i in range(len(files)):
            try:
                path = files[i]
                im = self.resize(path, (400, 400))
                im = self.L(im)
                im = self.narrow(im, ratio = 10)
                im = im.resize((320, 320))
                im = np.array(im)
        #         im, _ = self.histeq(im)
                im = self.imagenorm(im)
                _, name = os.path.split(path)
                self.saveImage(im * 125, os.path.join(targetFolder, name))
            except Exception as e:
                pass
      # 综合处理图片
    def resizeAndSaveImage(self, sourceFolder, targetFolder, shape = (320, 320)):
        self.createFolder(targetFolder)
        files = self.getImageFiles(sourceFolder)
        for i in range(len(files)):
            try:
                path = files[i]
                im = self.resize(path, shape)
                im = np.array(im)
                _, name = os.path.split(path)
                self.saveImage(im, os.path.join(targetFolder, name))
            except Exception as e:
                pass
    
   # 读取图片数据
    def readXData(self, folder, shape = (320, 320)):
        files = self.getImageFiles(folder)
        imagesize = len(files)
        newshape = list(shape)
        newshape.insert(0, imagesize)
        newshape = tuple(newshape)
        Xdata = np.ndarray(newshape, dtype = 'uint8')
        for i in range(len(files)):
            path = files[i]
            im = self.readImage(path)
#             Xdata[i] = self.imagenorm(im)
            Xdata[i] = im
        return Xdata
    
    # 读取标签数据
    def readYData(self, root, folder, excelPath, vocobsize = 7):
        files = self.getImageFiles(folder)
        imagesize = len(files)
        df = read_excel(os.path.join(root,excelPath))
        licenseChs = df.to_dict()['chs']
        licenseChs = { str(v):k for k, v in licenseChs.items()}
        Ydata = np.zeros((imagesize, vocobsize), dtype = 'int8')
        for i in range(len(files)):
            path = files[i]
            _, name = os.path.split(path)
            name, _ = os.path.splitext(name)
            namechs = list(name)
            for j in range(len(namechs)):
                Ydata[i, j] = int(licenseChs[namechs[j]])
        return Ydata
    
      # 保存数据npy
    def save(self, root, Xfilename, Yfilename, X, Y):
        Xpath = os.path.join(root, Xfilename)
        Ypath = os.path.join(root, Yfilename)
        np.save(Xpath, X)
        np.save(Ypath, Y)
        
    # 加载数据
    def load(self, root, Xfilename, Yfilename):
        Xpath = os.path.join(root, Xfilename)
        Ypath = os.path.join(root, Yfilename)
        X = np.load(Xpath)
        Y = np.load(Ypath)
        return X, Y
    
    #读取index->chs
    def readLabels(self, root, excelPath):
        excelPath = os.path.join(root, excelPath)
        df = read_excel(excelPath)
        licenseChs = df.to_dict()['chs']
        licenseChs = [ str(v) for k, v in licenseChs.items()]
        return licenseChs
    
    #根据下标获取车牌
    def getLabel(self, root, filename, y):
        chs = self.readLabels(root, filename)
        result = []
        for i in range(len(y)):
                result.append(chs[int(y[i])])
        return ''.join(result)
    
    '''unzip file'''
    def unzip(self, sourcepath, extractpath):
        with zipfile.ZipFile(sourcepath) as testzip:
            testzip.extractall(extractpath)
    
    '''from index get label'''
    def getlistargmax(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            outputs.append(np.argmax(inputs[i]))
        return outputs
import cv2
img = cv2.imread(r"F:\ASUS\Desktop\Densenet-Tensorflow-master\MNIST\a.png")
img2 = cv2.resize(img, (28,28))
cv2.imwrite(r"F:\ASUS\Desktop\Densenet-Tensorflow-master\MNIST\b.png", img2)
b2 =  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/255
cv2.imshow('b2', b2)#######################################演示图像用imshow
cv2.waitKey()##############################################等待键盘响应
cv2.destroyAllWindows()##################################键盘有任何动作退出

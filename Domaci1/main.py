import math
import sys
import numpy as np
import numpy.linalg as la
from PIL import Image

from PyQt5.QtCore import QPoint, pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QMainWindow, QLineEdit
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QPalette, QColor


def makePicture(P, outName):
    myImage = Image.open('building.jpg')
    myImage = myImage.resize((int(myImage.size[0]), int(myImage.size[1])))
    width, height = myImage.size

    F = la.inv(P)

    newImage = Image.new('RGB', (width, height), color=1)
    for i in range(width):
        for j in range(height):
            nc = F @ np.array([i, j, 1], dtype=np.int32).T

            # nc = P @ np.array([i, j, 1], dtype=np.int32)
            nx = nc[0, 0]
            ny = nc[0, 1]
            nz = nc[0, 2]
            if nz != 0:
                nx = int(nx / nz)
                ny = int(ny / nz)

            if nz != 0 and int(nx) < width and int(ny) < height and int(nx) >= 0 and int(ny) >= 0:
                n = 1
                r, g, b = myImage.getpixel((nx, ny))

                r = int(r / n)
                g = int(g / n)
                b = int(b / n)

                newImage.putpixel((i, j), (r, g, b))
    newImage.save("./{}.jpg".format(outName))
    newImage.show()


class Example(QMainWindow):
    def __init__(self):
        super(Example, self).__init__()


        self.imageWidth = 0
        self.imageHeight = 0

        self.x1p = QLineEdit(self)
        self.x2p = QLineEdit(self)
        self.x3p = QLineEdit(self)
        self.x4p = QLineEdit(self)
        self.x5p = QLineEdit(self)
        self.x6p = QLineEdit(self)
        self.x7p = QLineEdit(self)
        self.x8p = QLineEdit(self)
        self.x9p = QLineEdit(self)
        self.x10p = QLineEdit(self)
        self.x11p = QLineEdit(self)
        self.x12p = QLineEdit(self)

        self.x1 = QLineEdit(self)
        self.x2 = QLineEdit(self)
        self.x3 = QLineEdit(self)
        self.x4 = QLineEdit(self)
        self.x5 = QLineEdit(self)
        self.x6 = QLineEdit(self)
        self.x7 = QLineEdit(self)
        self.x8 = QLineEdit(self)
        self.x9 = QLineEdit(self)
        self.x10 = QLineEdit(self)
        self.x11 = QLineEdit(self)
        self.x12 = QLineEdit(self)


        self.num = QLabel(self)
        self.boxes = [self.x1, self.x2,
                      self.x3, self.x4,
                      self.x5, self.x6,
                      self.x7, self.x8,
                      self.x9, self.x10,
                      self.x11, self.x12]

        self.dots = []
        self.numDots = 0
        self.images = []

        self.initUI()

    def mouseReleaseEvent(self, QMouseEvent):
        cursor = QtGui.QCursor()
        x = cursor.pos().x() - self.pos().x()
        y = cursor.pos().y() - self.pos().y()
        if x < self.imageWidth and y < self.imageHeight and self.numDots < 12:
            self.boxes[self.numDots].setText("({}:{}:1)".format(x, y))
            self.dots.append([x, y, 1])
            self.numDots += 1
            self.num.setText("{}".format(self.numDots))

    @pyqtSlot()
    def naiveAlg(self):
        print("naive")

        a = np.array([self.dots[0][0], self.dots[0][1], self.dots[0][2]])
        b = np.array([self.dots[1][0], self.dots[1][1], self.dots[1][2]])
        c = np.array([self.dots[2][0], self.dots[2][1], self.dots[2][2]])
        d = np.array([self.dots[3][0], self.dots[3][1], self.dots[3][2]])

        ap = self.x1p.text().split(":")
        ap[0] = ap[0][1:]
        ap[2] = ap[2][:-1]

        ap = np.array([float(x) for x in ap])

        bp = self.x2p.text().split(":")
        bp[0] = bp[0][1:]
        bp[2] = bp[2][:-1]

        bp = np.array([float(x) for x in bp])

        cp = self.x3p.text().split(":")
        cp[0] = cp[0][1:]
        cp[2] = cp[2][:-1]

        cp = np.array([float(x) for x in cp])

        dp = self.x4p.text().split(":")
        dp[0] = dp[0][1:]
        dp[2] = dp[2][:-1]

        dp = np.array([float(x) for x in dp])

        alpha = la.det([d, b, c])
        beta = la.det([a, d, c])
        gamma = la.det([a, b, d])

        a_new = alpha * a
        b_new = beta * b
        c_new = gamma * c

        G = np.matrix([[a_new[0], b_new[0], c_new[0]],
                       [a_new[1], b_new[1], c_new[1]],
                       [a_new[2], b_new[2], c_new[2]]])

        alphaP = la.det([dp, bp, cp])
        betaP = la.det([ap, dp, cp])
        gammaP = la.det([ap, bp, dp])

        ap_new = alphaP * ap
        bp_new = betaP * bp
        cp_new = gammaP * cp

        F = np.matrix([[ap_new[0], bp_new[0], cp_new[0]],
                       [ap_new[1], bp_new[1], cp_new[1]],
                       [ap_new[2], bp_new[2], cp_new[2]]])

        P = F * (la.inv(G))
        print(P)
        makePicture(P, "Naive")

    @pyqtSlot()
    def dltAlg(self):
        print("dlt")

        dots = []

        for i in range(self.numDots):
            dots.append(np.array([self.dots[i][0], self.dots[i][1], self.dots[i][2]]))

        images = []

        for i in range(self.numDots):
            tmp = self.images[i].text().split(":")

            tmp[0] = tmp[0][1:]
            tmp[2] = tmp[2][:-1]

            images.append(np.array([float(x) for x in tmp]))

        matrixM = []

        for i in range(self.numDots):
            X = dots[i]
            Xp = images[i]

            m0 = [0, 0, 0, -Xp[2] * X[0], -Xp[2] * X[1], -Xp[2] * X[2], Xp[1] * X[0], Xp[1] * X[1], Xp[1] * X[2]]
            m1 = [Xp[2] * X[0], Xp[2] * X[1], Xp[2] * X[2], 0, 0, 0, -Xp[0] * X[0], -Xp[0] * X[1], -Xp[0] * X[2]]

            matrixM.append(m0)
            matrixM.append(m1)

        A = np.matrix(matrixM)

        _, _, vt = la.svd(A, full_matrices=True, compute_uv=True)

        v = vt.T
        P = v[:, -1]
        P = P.reshape(3, 3)

        print(P)

        makePicture(P, "DLT")

    @pyqtSlot()
    def normalizedDlt(self):
        print("normalized")

        n = self.numDots

        G = [0, 0]
        Gp = [0, 0]

        l = 0
        lp = 0

        for i in range(n):
            A = self.dots[i]
            A1 = A[0]/A[2]
            A2 = A[1]/A[2]

            G = [G[0]+A1, G[1]+A2]

            tmp = self.images[i].text().split(":")
            tmp[0] = tmp[0][1:]
            tmp[2] = tmp[2][:-1]

            B = [float(x) for x in tmp]
            B1 = B[0]/B[2]
            B2 = B[1]/B[2]

            Gp = [Gp[0]+B1, Gp[1]+B2]

            l += math.sqrt(A1*A1+A2*A2)
            lp += math.sqrt(B1*B1+B2*B2)

        G = [G[0]/n, G[1]/n]
        Gp = [Gp[0]/n, Gp[1]/n]

        k = math.sqrt(2)/(l/n)
        kp = math.sqrt(2)/(lp/n)

        TrG = [[1, 0, -G[0]],
               [0, 1, -G[1]],
               [0, 0, 1]]

        TrGp = [[1, 0, -Gp[0]],
                [0, 1, -Gp[1]],
                [0, 0, 1]]

        S = [[k, 0, 0],
             [0, k, 0],
             [0, 0, 1]]

        Sp = [[kp, 0, 0],
              [0, kp, 0],
              [0, 0, 1]]

        T = np.matrix(S) @ np.matrix(TrG)
        Tp = np.matrix(Sp) @ np.matrix(TrGp)


        dots = []

        for i in range(self.numDots):
            dots.append(np.array([self.dots[i][0], self.dots[i][1], self.dots[i][2]]))

        images = []

        for i in range(self.numDots):
            tmp = self.images[i].text().split(":")

            tmp[0] = tmp[0][1:]
            tmp[2] = tmp[2][:-1]

            images.append(np.array([float(x) for x in tmp]))

        dots = [T @ np.array(x) for x in dots]
        images = [Tp @ np.array(x) for x in images]

        matrixM = []

        for i in range(self.numDots):
            X = dots[i]
            Xp = images[i]

            m0 = [0, 0, 0, -Xp[0, 2] * X[0, 0], -Xp[0, 2] * X[0, 1], -Xp[0, 2] * X[0, 2], Xp[0, 1] * X[0, 0], Xp[0, 1] * X[0, 1], Xp[0, 1] * X[0, 2]]
            m1 = [Xp[0, 2] * X[0, 0], Xp[0, 2] * X[0, 1], Xp[0, 2] * X[0, 2], 0, 0, 0, -Xp[0, 0] * X[0, 0], -Xp[0, 0] * X[0, 1], -Xp[0, 0] * X[0, 2]]

            matrixM.append(m0)
            matrixM.append(m1)

        A = np.matrix(matrixM)

        _, _, vt = la.svd(A, full_matrices=True, compute_uv=True)

        v = vt.T
        Pn = v[:, -1]
        Pn = Pn.reshape(3, 3)

        P = la.inv(Tp) @ Pn @ T

        print(P)
        makePicture(P, "NormalizedDLT")

    @pyqtSlot()
    def clear(self):
        self.num.setText("0")

        for text in self.boxes:
            text.setText("")

        for text in self.images:
            text.setText("")

        self.numDots = 0
        self.dots = []

    def initUI(self):

        label = QLabel(self)
        pixmap = QPixmap('building.jpg')
        width = pixmap.width()
        height = pixmap.height()
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        label.resize(width, height)
        label.setPixmap(pixmap)

        self.imageHeight = height
        self.imageWidth = width

        naive = QPushButton("Naive Algorithm", self)
        naive.resize(100, 30)
        naive.move(10, max(height + 10, 550))
        naive.clicked.connect(self.naiveAlg)
        naive.setToolTip('Run Naive algorithm')

        dlt = QPushButton("DLT Algorithm", self)
        dlt.resize(150, 30)
        dlt.move(120, max(height + 10, 550))
        dlt.clicked.connect(self.dltAlg)
        dlt.setToolTip('Run DLT algorithm')

        normalized = QPushButton("Normalized DLT", self)
        normalized.resize(110, 30)
        normalized.move(280, max(height + 10, 550))
        normalized.clicked.connect(self.normalizedDlt)
        normalized.setToolTip('Run Normalized DLT algorithm')

        clear = QPushButton("Clear", self)
        clear.resize(50, 30)
        clear.move(510, max(height + 10, 550))
        clear.clicked.connect(self.clear)
        clear.setToolTip('Clear the coordinates')

        xsLabel = QLabel(self)
        xsLabel.setText("Xs")
        xsLabel.move(width + 40, 5)

        xPsLabel = QLabel(self)
        xPsLabel.setText("X's")
        xPsLabel.move(width + 130, 5)

        self.x1.resize(80, 20)
        self.x1.move(width + 10, 30)
        self.x1.setDisabled(True)
        self.x1.setToolTip('(x1:x2:x3)')

        self.x2.resize(80, 20)
        self.x2.move(width + 10, 70)
        self.x2.setDisabled(True)
        self.x2.setToolTip('(x1:x2:x3)')

        self.x3.resize(80, 20)
        self.x3.move(width + 10, 110)
        self.x3.setDisabled(True)
        self.x3.setToolTip('(x1:x2:x3)')

        self.x4.resize(80, 20)
        self.x4.move(width + 10, 150)
        self.x4.setDisabled(True)
        self.x4.setToolTip('(x1:x2:x3)')

        self.x5.resize(80, 20)
        self.x5.move(width + 10, 190)
        self.x5.setDisabled(True)
        self.x5.setToolTip('(x1:x2:x3)')

        self.x6.resize(80, 20)
        self.x6.move(width + 10, 230)
        self.x6.setDisabled(True)
        self.x6.setToolTip('(x1:x2:x3)')

        self.x7.resize(80, 20)
        self.x7.move(width + 10, 270)
        self.x7.setDisabled(True)
        self.x7.setToolTip('(x1:x2:x3)')

        self.x8.resize(80, 20)
        self.x8.move(width + 10, 310)
        self.x8.setDisabled(True)
        self.x8.setToolTip('(x1:x2:x3)')

        self.x9.resize(80, 20)
        self.x9.move(width + 10, 350)
        self.x9.setDisabled(True)
        self.x9.setToolTip('(x1:x2:x3)')

        self.x10.resize(80, 20)
        self.x10.move(width + 10, 390)
        self.x10.setDisabled(True)
        self.x10.setToolTip('(x1:x2:x3)')

        self.x11.resize(80, 20)
        self.x11.move(width + 10, 430)
        self.x11.setDisabled(True)
        self.x11.setToolTip('(x1:x2:x3)')

        self.x12.resize(80, 20)
        self.x12.move(width + 10, 470)
        self.x12.setDisabled(True)
        self.x12.setToolTip('(x1:x2:x3)')

        self.x1p.resize(80, 20)
        self.x1p.move(width + 100, 30)
        self.x1p.setToolTip('(x1:x2:x3)')

        self.x2p.resize(80, 20)
        self.x2p.move(width + 100, 70)
        self.x2p.setToolTip('(x1:x2:x3)')

        self.x3p.resize(80, 20)
        self.x3p.move(width + 100, 110)
        self.x3p.setToolTip('(x1:x2:x3)')

        self.x4p.resize(80, 20)
        self.x4p.move(width + 100, 150)
        self.x4p.setToolTip('(x1:x2:x3)')

        self.x5p.resize(80, 20)
        self.x5p.move(width + 100, 190)
        self.x5p.setToolTip('(x1:x2:x3)')

        self.x6p.resize(80, 20)
        self.x6p.move(width + 100, 230)
        self.x6p.setToolTip('(x1:x2:x3)')

        self.x7p.resize(80, 20)
        self.x7p.move(width + 100, 270)
        self.x7p.setToolTip('(x1:x2:x3)')

        self.x8p.resize(80, 20)
        self.x8p.move(width + 100, 310)
        self.x8p.setToolTip('(x1:x2:x3)')

        self.x9p.resize(80, 20)
        self.x9p.move(width + 100, 350)
        self.x9p.setToolTip('(x1:x2:x3)')

        self.x10p.resize(80, 20)
        self.x10p.move(width + 100, 390)
        self.x10p.setToolTip('(x1:x2:x3)')

        self.x11p.resize(80, 20)
        self.x11p.move(width + 100, 430)
        self.x11p.setToolTip('(x1:x2:x3)')

        self.x12p.resize(80, 20)
        self.x12p.move(width + 100, 470)
        self.x12p.setToolTip('(x1:x2:x3)')

        self.images = [self.x1p, self.x2p,
                       self.x3p, self.x4p,
                       self.x5p, self.x6p,
                       self.x7p, self.x8p,
                       self.x9p, self.x10p,
                       self.x11p, self.x12p]

        tackeLab = QLabel(self)
        tackeLab.setText("Izabrano tacaka: ")
        tackeLab.move(self.imageWidth + 10, 500)
        self.num.setText("0")
        self.num.move(self.imageWidth + 130, 500)

        self.setGeometry(0, 0, max(width + 200, 650), max(height + 50, 600))
        self.move(100, 50)
        self.setFixedSize(self.size())
        self.setWindowTitle('Otklanjanje projektivne distorzije')
        self.setWindowFlags(self.windowFlags())
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = Example()

    app.setStyle("Fusion")

    lightBlue = QColor(0, 191, 255)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, lightBlue)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, lightBlue)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, lightBlue)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, lightBlue)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

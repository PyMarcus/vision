import cv2
import numpy as np
from matplotlib import pyplot as plt


class ComputationalVision:
    """
    Visão computacional é a ciência e tecnologia
    das máquinas que enxergam, ou seja, é a arte
    de extrair informações de imagens e objetos multidimencionais

    -------------------------------------------------------------------------
                                    A imagem
    Cada imagem, é uma matriz de 3 dimensões (se for colorida),
    se nao, há 2 dimensoes
    Cada ponto da imagem, é um pixel, que é equivalente a 8bits (uint8)
    Assim, cada matriz de cores é combinada formando milhões de possibilidades
    de cores 256^3

    --> Quanto mais proximos de 255px,na cor, 0xFF, mais claro é, pois 255, 255, 255 é branco
    e quanto mais perto de zero, mais escuro é
    -------------------------------------------------------------------------
                             Coordenadas de uma imagem
    Sendo uma imagem, matriz, por exemplo, 200x300, implica em 199 linhas (pixels) por
    299 colunas (pixels), assim, pode-se alterar cada pixel da imagem.
    """

    RED = (0, 0, 0xFF)
    GREEN = (0, 0xFF, 0)
    BLUE = (0xFF, 0, 0)

    def __init__(self) -> None:
        self.__base: str = "images/"
        self.__image01: str = self.__base + "image01.jpg"
        self.__image02: str = self.__base + "image02.jpg"
        self.__test: str = self.__base + "test.jpg"

    def view_image(self, image) -> None:
        cv2.imshow("Imagem", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_and_get_info_image(self) -> None:
        print(self.__image01)
        image = cv2.imread(self.__image01)
        print(f"HEIGHT of {self.__image01}: {image.shape[0]}px")
        print(f"WIDTH of {self.__image01}: {image.shape[1]}px")
        print(f"CHANNELS (RGB) {image.shape[2]}")
        cv2.imshow("Window name", image)  # exibe a imagem
        cv2.waitKey(0)  # aguarda alguma tecla
        # cv2.imwrite("name.jpg",image)

    def coordinate_of_image(self) -> None:
        image = cv2.imread(self.__image01)
        b, g, r = image[0, 0]  # linha 0 coluna 0, pixel (0, 0), contem tupla de bgr
        print(f"RED {r} - GREEN {g} - BLUE {b}")
        for line_pixel in range(len(image)):
            for column_pixel in range(len(image)):
                # print(f"{line_pixel, column_pixel}")
                image[line_pixel:line_pixel + 10, column_pixel:column_pixel + 10] = image[
                    line_pixel % 2, column_pixel % 2]
            # print()
        cv2.imshow("Modify image", image)
        cv2.waitKey(0)

    def one_point_modify(self) -> None:
        image = cv2.imread(self.__image02)
        image[300: 301, 300: 301] = (0, 0, 0xFF)
        cv2.imshow("test", image)
        cv2.waitKey(0)

    def create_a_circle(self) -> None:
        image = cv2.imread(self.__test)
        for radio in range(0, 200, 10):
            cv2.circle(image, (500, 500), radio, (0xFF, 0xFF, 0xFF))
        cv2.imshow("circle", image)
        cv2.waitKey(0)

    def write_text_on_image(self) -> None:
        image = cv2.imread(self.__test)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Marcus', (50, 50), font, 2, (0, 0, 0xFF), 2, cv2.LINE_AA)
        self.view_image(image)

    def transformations(self) -> None:
        """Transformar implica em recortar, redimencionar etc"""

        # recorte
        image = cv2.imread(self.__test)
        cut = image[0: 100, 0: 100]
        # self.view_image()

        # redimencionar
        image = cv2.imread(self.__test)
        cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
        # self.view_image(resize)

    def mask(self) -> None:
        # imagem onde há apenas pixels binários (pretos ou brancos)
        img = cv2.imread(self.__test)
        m = np.zeros(img.shape[:2], dtype="uint8")  # cria matriz de zeros com o tamanho da imagem
        cX, cY = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.circle(m, (cX, cY), 100, 255, -1)  # cria um circulo na imagem
        img_with_mask = cv2.bitwise_and(img, img, mask=m)  #
        self.view_image(img_with_mask)

    def histogram(self) -> None:
        # gráfico que mostra a distribuição e concentração de pixels da imagem
        image = cv2.imread(self.__image01)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converte para P&B (cinza)
        h = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("RGB")
        plt.ylabel("Pixels")
        plt.plot(h)
        plt.xlim([0, 256])
        plt.show()

    def smoothing(self) -> None:
        # suavização da imagem,isto é,efeito blur de desfoque, ideal para localizar objetos
        image = cv2.imread(self.__test)
        image = cv2.blur(image, (5, 5))  # numeros impares para manter o centor
        self.view_image(image)

    def smoothing_bilateral_filter(self) -> None:
        # mantem as bordas ao aplicar suavizacao
        image = cv2.imread(self.__image02)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.bilateralFilter(image, 3, 21, 21)
        self.view_image(image)


if __name__ == '__main__':
    cv = ComputationalVision()
    cv.smoothing_bilateral_filter()

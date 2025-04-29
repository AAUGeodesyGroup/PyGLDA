import numpy as np


class PCA_EOF:

    def __init__(self):
        self.__eigValue = None
        self.__PC = None
        self.__EOF = None

        pass

    def setField(self, X, demean=False):
        """
        Notice: the dimension 0 denotes the samples, and the dimension 1 denotes the spatial point.
        Notice: the number of features is assumed to be less than that of the samples.
        :param X: target spatial-temporal fields; two-dimension [N,M]: N time samples, M grid points
        :return:
        """

        '''Centered'''
        if demean:
            mean = np.mean(X, axis=0)
            X = X - mean
        '''PCA'''
        shp = np.shape(X)

        if shp[0] < shp[1]:

            cov = np.dot(X, X.T)

            eigvalue, eigvector = np.linalg.eig(cov)

            self.__eigValue = eigvalue

            eigvalue[eigvalue < 0] = np.abs(eigvalue[eigvalue < 0])

            itri = np.diag(np.sqrt(eigvalue))

            # itri = np.c_[itri, np.zeros((shp[0], shp[1] - shp[0]))]

            self.__PC = np.dot(eigvector, itri)

            V = np.dot(eigvector.T, X)

            a = np.dot(itri.T, np.diag(1. / eigvalue))

            EOF = np.dot(a, V)

            self.__EOF = EOF.T

        else:

            cov = np.dot(X.T, X)

            eigvalue, eigvector = np.linalg.eig(cov)

            self.__eigValue = eigvalue

            # eigvalue[eigvalue < 0] = np.abs(eigvalue[eigvalue < 0])
            eigvalue[eigvalue < 0] = 0

            self.__EOF = eigvector

            self.__PC = np.dot(X, self.__EOF.copy())

        '''flip signs to provide consistent results:'''
        ind = self.__PC[0, :] < 0
        self.__PC[:, ind] = self.__PC[:, ind] * (-1)
        self.__EOF[:, ind] = self.__EOF[:, ind] * (-1)

        return self

    def getPCs(self, topN):
        """
        get PCs time-series
        :param topN: Top N sorted by variance
        :return:
        """
        sorted_indices = np.argsort(-self.__eigValue)

        return self.__PC[:, sorted_indices[0:topN]].T

    def getEOFs(self, topN):
        """
        get EOFs, spatial distribution
        :param topN: Top N sorted by variance
        :return:
        """
        sorted_indices = np.argsort(-self.__eigValue)
        return self.__EOF[:, sorted_indices[0:topN]].T

    def getVarPercent(self, topN):
        """
        get the percent of top N modes.
        :param topN:
        :return:
        """
        sorted_indices = np.argsort(-self.__eigValue)

        tot = np.sum(self.__eigValue)

        return self.__eigValue[sorted_indices[0:topN]] / tot


def demo():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    '''Centered'''
    mean = np.mean(X, axis=0)
    # X = X - mean
    '''PCA'''
    shp = np.shape(X)

    cov = np.dot(X, X.T)

    eigvalue, eigvector = np.linalg.eig(cov)
    eigvalue1, eigvector1 = np.linalg.eig(cov / 2)

    cov1 = np.dot(X.T, X)
    eigvalue2, eigvector2 = np.linalg.eig(cov1)
    eigvalue3, eigvector3 = np.linalg.eig(cov1 / 3)

    s, u, v = np.linalg.svd(X)

    pca = PCA_EOF().setField(X)
    eof = pca.getEOFs(2)

    pass


if __name__ == '__main__':
    demo()
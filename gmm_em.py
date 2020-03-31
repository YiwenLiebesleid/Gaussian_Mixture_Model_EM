import pandas as pd
import numpy as np
import time

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

############################################
### Setup
############################################

def readData(filename, cluster=False):
    df = pd.read_csv(filename, header=None)
    clusters = None
    if cluster:
        clusters = df.pop(2)
    points = np.array(df)
    return points, clusters


class GaussianMMEM:
    def __init__(self, points, num_clusters, iterations, epsilon=None):
        self.points = points
        self.mu = None
        self.pi = None
        self.sigma = None
        self.epsilon = epsilon
        self.num_clusters = num_clusters
        self.iterations = iterations

        # log-likelihood per iteration
        self.log_likelihoods = []
        self.bic = float('-inf')

    def run(self, display=True):
        # Helps prevent Gaussian collapse
        self.reg_cov = 1e-6 * np.identity(len(self.points[0]))

        x_grid, y_grid = np.meshgrid(np.sort(self.points[:, 0]), np.sort(self.points[:, 1]))
        self.points_XY = np.array([x_grid.flatten(), y_grid.flatten()]).T

        self.initialize_gaussians(display)

        oldlikelihood = float('-inf')
        k = self.iterations
        for i in range(self.iterations):
            probabilities, likelihood = self.expectation()

            self.maximization(probabilities)
            #
            if i > 0:
                self.log_likelihoods.append(likelihood)
                if likelihood - oldlikelihood < 0.00001:
                    break

            oldlikelihood = likelihood

        probabilities, likelihood = self.expectation()
        self.calculate_bic()
        self.log_likelihoods.append(likelihood)
        if display:
            self.display_log_likelihoods()
            self.display_state("Final State")

    def calculate_bic(self):
        # self.bic = self.log_likelihoods[-1] - 0.5 * (len(self.points[0]) + 1) * self.num_clusters * np.log(len(self.points))
        self.bic = self.log_likelihoods[-1] - (len(self.points[0]) + 1) * self.num_clusters * np.log(len(self.points))

    def display_log_likelihoods(self):
        fig = plt.figure(figsize=(10, 10))
        subp = fig.add_subplot(111)
        subp.set_title('Log-likelihood')
        subp.plot(range(0, len(self.log_likelihoods), 1), self.log_likelihoods)
        # fig.show()
        fig.savefig("log-likelihood.jpg")

    def calc_log_likelihood(self):
        gaussians = []
        for i, pi in enumerate(self.pi):
            gaussians.append(multivariate_normal(self.mu[i], self.sigma[i]))
        tot = 0
        for p in self.points:
            ptot = 0
            for pi, gauss in zip(self.pi, gaussians):
                ptot += pi * gauss.pdf(p)
            tot = np.log(ptot)
        return tot

    def expectation(self):
        r_ic = np.zeros((len(self.points), self.num_clusters))

        # Compute probabilities for each point and total probabilities
        total_probs = np.zeros(len(self.points), dtype='float')
        for mu, cov, pi, c in zip(self.mu, self.sigma, self.pi, range(self.num_clusters)):
            cov += self.reg_cov
            multi_norm = multivariate_normal(mu, cov)
            probs = multi_norm.pdf(self.points) * pi
            total_probs += probs
            r_ic[:, c] = probs

        # log likelihood
        cprobs = np.sum(r_ic, axis=1)
        p = np.log(cprobs[:])
        likelihood = np.sum(p)

        # Normalize
        for c in range(self.num_clusters):
            r_ic[:, c] = r_ic[:, c] / total_probs

        return r_ic, likelihood

    def maximization(self, r_ic):
        # compute gaussian parameters
        self.mu = []
        self.sigma = []
        self.pi = []

        for c in range(self.num_clusters):
            # Can think of as the fraction of points allocated to cluster
            m_c = np.sum(r_ic[:, c], axis=0)
            mu_c = (1 / m_c) * np.sum(self.points * r_ic[:, c].reshape(len(self.points), 1), axis=0)
            self.mu.append(mu_c)

            cov = ((1 / m_c) * np.dot(
                (np.array(r_ic[:, c]).reshape(len(self.points), 1) * (self.points - mu_c)).T,
                (self.points - mu_c))) + self.reg_cov
            self.sigma.append(cov)

            self.pi.append(m_c / np.sum(r_ic))

    def initialize_gaussians(self, display):
        self.mu = np.random.randint(min(self.points[:, 0]), max(self.points[:, 0]),
                                    size=(self.num_clusters, len(self.points[0])))
        self.pi = np.ones(self.num_clusters) / self.num_clusters
        self.sigma = np.ones([self.num_clusters, len(self.points[0]), len(self.points[0])])

        for dim in range(self.num_clusters):
            np.fill_diagonal(self.sigma[dim], 5)

        if display:
            self.display_state("Initial State")

    def display_state(self, title):
        fig = plt.figure(figsize=(15, 15))
        subp = fig.add_subplot(111)
        subp.scatter(self.points[:, 0], self.points[:, 1])
        subp.set_title(title)

        for mu, cov in zip(self.mu, self.sigma):
            cov += self.reg_cov
            multi_norm = multivariate_normal(mu, cov)
            subp.contour(np.sort(self.points[:, 0]), np.sort(self.points[:, 1]),
                         multi_norm.pdf(self.points_XY).reshape(len(self.points), len(self.points)), colors='black',
                         alpha=0.3)
            subp.scatter(mu[0], mu[1], c='grey', zorder=10, s=100)
        # fig.show()
        fig.savefig(title + ".jpg")


def select_model(points, K):
    global start_time
    start_time = time.time()
    if K != 0:
        best_result = None
        while True:
            if time.time() - start_time > 9.8:  break
            try:
                GMM = GaussianMMEM(points, K, 100)
                GMM.run(display=False)
                if best_result is None:
                    best_result = GMM
                else:
                    if GMM.bic > best_result.bic:   best_result = GMM
            except ValueError:
                continue
        print("time: " + str(time.time() - start_time))
        print("Best BIC: " + str(best_result.bic), " Best LL: " + str(best_result.log_likelihoods[-1]))
        best_result.display_log_likelihoods()
        best_result.display_state("Final State")
    else:
        BIC = []
        best_bic = float('-inf')
        best_result = None
        for k in range(2, 20):
            if time.time() - start_time > 9.8:  break
            while True:
                try:
                    GMM = GaussianMMEM(points, k, 100)
                    GMM.run(display=False)
                    BIC.append(GMM.bic)
                    break
                except ValueError:
                    continue
            if BIC[-1] <= best_bic:  continue
            else:
                best_bic = BIC[-1]
                best_result = GMM
        best_K = np.argmax(BIC) + 2
        print("Time: " + str(time.time() - start_time))
        print("Best K: " + str(best_K), " Best BIC: " + str(best_bic), " Best LL: " + str(best_result.log_likelihoods[-1]))
        best_result.display_log_likelihoods()
        best_result.display_state("Final State")
    return best_result

if __name__ == '__main__':

    command = input()
    filename, K = command.split(' ')[1:]
    K = int(K)
    points, cluster_labels = readData(filename)
    p = points[:500]
    model = select_model(points, K)
    print("Cluster Centers: ",np.array(model.mu))

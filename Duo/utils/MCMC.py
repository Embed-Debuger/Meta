import numpy as np


def Update(corpus):
    for index in range(len(corpus)):
        corpus[index][1] = np.random.rand()
    # corpus[0][1] = 0.3
    # corpus[1][1] = 0.25
    # corpus[2][1] = 0.66
    # corpus[3][1] = 0.12
    # corpus[4][1] = 0.37


def Search(cur, corpus):
    for index in range(0, len(corpus)):
        if (cur == corpus[index][0]):
            return index


def plot_mcmc(corpus, p):
    cur = np.random.randint(0, len(corpus) - 1)
    cur = corpus[cur][0]
    states = [cur]
    for i in range(500):
        corpus.sort(key=lambda num: num[1], reverse=True)
        cur = Search(cur, corpus)
        next = np.random.randint(0, len(corpus) - 1)
        u = corpus[next][1]
        if u >= min(pow(1 - p, next - cur), 1):
            states.append(corpus[next][0])
            # if len(states) == 6:
            #     break
            cur = corpus[next][0]
        else:
            cur = corpus[cur][0]

        Update(corpus)
        print(states)
    # return states


if __name__ == '__main__':
    corpus = [['m1', 0.3], ['m2', 0.25], ['m3', 0.66], ['m4', 0.12], ['m5', 0.37]]
    print(plot_mcmc(corpus, 0.2))

import pong
import numpy as np
from scipy.special import softmax
import copy

def relu(x):
    return np.maximum(0, x)


def new_network(numinputs, hiddenlayer, outputs, seed):
    rng = np.random.default_rng(seed)
    l1weights = np.zeros((numinputs, hiddenlayer))
    l1biases = np.zeros(hiddenlayer)
    for input in range(numinputs):
        for hidden in range(hiddenlayer):
            l1weights[input][hidden] = rng.normal(loc=0, scale=1, size=None)
    for hidden in range(hiddenlayer):
        l1biases[hidden] =  rng.normal(loc=0, scale=1, size=None)

    l2weights = np.zeros((hiddenlayer, outputs))
    l2biases = np.zeros(outputs)
    for hidden in range(hiddenlayer):
        for out in range(outputs):
            l2weights[hidden][out] = rng.normal(loc=0, scale=1, size=None)

    for out in range(outputs):
        l2biases[out] = rng.normal(loc=0, scale=1, size=None)
    return {
        "l1weights" : l1weights,
        "l1biases" : l1biases,
        "l2weights" : l2weights,
        "l2biases" : l2biases
    }
    

def forward_pass(game, network):
    assert len(list(network['l1weights'])) == 6
    inputactivations = np.zeros(6)
    gamekeys = [
            "ballx",
            "bally",    
            "ballspeed",
            "ballangle",
            "playerpaddle",
            "opponentpaddle"]

    for _ in range(len(gamekeys)):
        inputactivations[_] = game[gamekeys[_]]

    inputactivations[0] /= game["width"]
    inputactivations[1] /= game["height"]
    inputactivations[4] /= game["width"]
    inputactivations[5] /= game["width"]

    secondlayer = relu(network["l1weights"].T @ inputactivations + network["l1biases"])
    return softmax(network["l2weights"].T @ secondlayer + network["l2biases"])

def playgame(nn1, nn2, seed=2):
    rng = np.random.default_rng(seed)
    currgame = pong.new_game(720, 1080, rng.integers(low=0, high=1000))
    actions = ["left", None, "right"]

    for _ in range(6000):
        nn1move = rng.choice(actions, p=forward_pass(currgame, nn1))
        nn2move = rng.choice(actions, p=forward_pass(pong.flipsides(currgame), nn2))
        currgame["playeraction"] = nn1move
        currgame["opponentaction"] = nn2move
        pong.frame_step(currgame)
        if currgame["state"] == "won":

            return (3*_, 0.5*_)
        if currgame["state"] == "lost":

            return (0.5*_, 3*_)
    return (6000, 6000)


def generate(num, seed):
    rng = np.random.default_rng(seed)
    nnseeds = rng.integers(low=0, high=10000, size=num)
    
    output = np.array([new_network(6, 7, 3, nnseed) for nnseed in nnseeds])

    return output

def mutate(nn, rate, seed):
    newnn = copy.deepcopy(nn)
    rng = np.random.default_rng(seed)
    for row in range(len(nn["l1weights"])):
        for val in range(len(nn["l1weights"][row])):
            if rng.uniform(0, 1) <= rate:
                newnn["l1weights"][row][val] += rng.normal(loc=0, scale=0.1, size=None)
    for val in range(len(nn["l1biases"])):
        if rng.uniform(0, 1) <= rate:
            newnn["l1biases"][val] += rng.normal(loc=0, scale=0.1, size=None)

    for row in range(len(nn["l2weights"])):
        for val in range(len(nn["l2weights"][row])):
            if rng.uniform(0, 1) <= rate:
                newnn["l2weights"][row][val] += rng.normal(loc=0, scale=0.1, size=None)
    for val in range(len(nn["l2biases"])):
        if rng.uniform(0, 1) <= rate:
            newnn["l2biases"][val] += rng.normal(loc=0, scale=0.1, size=None)
    return newnn

def generation_increment(nns, seed):
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(nns))
    for nn1 in range(len(nns)):
        for nn2 in range(len(nns)):
            points = playgame(nns[nn1], nns[nn2], rng.integers(low=0, high=1000))
            scores[nn1] += points[0]
            scores[nn2] += points[1]


    nextgen = []
    nextgen.append(nns[np.argmax(scores)])
    #top parent from previous generation continues to live in next generation

    scores = softmax(scores**(1/5))
    for _ in range(len(nns) - 1):
        newnn = mutate(rng.choice(nns, p=scores), 0.1, rng.integers(low=0, high=1000))
        nextgen.append(newnn)



    return nextgen


if __name__ == "__main__":
    curr = generate(40, 0)
    for _ in range(10000):
        curr = generation_increment(curr, 12398)
        if _ %10 == 0:
            print(str(_) + " completed generations")
            print(curr[0])
        if _%100 == 0:
            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(str(curr))
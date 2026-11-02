import pong
import numpy as np

import copy

def relu(x):
    return np.maximum(0, x)

def ball_tracker(game):
    if game["playerpaddle"] < game["ballx"]:
        return np.array([0, 0, 1])
    elif game["playerpaddle"] > game["ballx"]:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])



def new_network(numinputs, hiddenlayer1, hiddenlayer2, outputs, seed):
    rng = np.random.default_rng(seed)
    l1weights = np.zeros((numinputs, hiddenlayer1))
    l1biases = np.zeros(hiddenlayer1)
    for input in range(numinputs):
        for hidden in range(hiddenlayer1):
            l1weights[input][hidden] = rng.normal(loc=0, scale=1, size=None)
    for hidden in range(hiddenlayer1):
        l1biases[hidden] =  rng.normal(loc=0, scale=1, size=None)

    l2weights = np.zeros((hiddenlayer1, hiddenlayer2))
    l2biases = np.zeros(hiddenlayer2)
    for hidden1 in range(hiddenlayer1):
        for hidden2 in range(hiddenlayer2):
            l2weights[hidden1][hidden2] = rng.normal(loc=0, scale=1, size=None)

    for hidden2 in range(hiddenlayer2):
        l2biases[hidden2] = rng.normal(loc=0, scale=1, size=None)

    l3weights = np.zeros((hiddenlayer2, outputs))
    l3biases = np.zeros(outputs)

    for hidden2 in range(hiddenlayer2):
        for out in range(outputs):
            l3weights[hidden2][out] = rng.normal(loc=0, scale=1, size=None)

    for out in range(outputs):
        l3biases[out] = rng.normal(loc=0, scale=1, size=None)

    return {
        "l1weights" : l1weights,
        "l1biases" : l1biases,
        "l2weights" : l2weights,
        "l2biases" : l2biases,
        "l3weights" : l3weights,
        "l3biases" : l3biases
    }
    

def forward_pass(game, network):
    assert len(list(network['l1weights'])) == 7
    inputactivations = np.zeros(7)
    gamekeys = [
            "ballx",
            "bally",    
            "ballspeed",
            "ballangle",
            "ballangle",
            "playerpaddle",
            "opponentpaddle"]

    for _ in range(len(gamekeys)):
        inputactivations[_] = game[gamekeys[_]]

    inputactivations[0] /= game["width"]
    inputactivations[1] /= game["height"]
    inputactivations[2] -= 5
    inputactivations[2] /= 5
    inputactivations[3] = np.sin(inputactivations[3])
    inputactivations[4] = np.cos(inputactivations[4])

    inputactivations[5] /= game["width"]
    inputactivations[6] /= game["width"]

    secondlayer = relu(network["l1weights"].T @ inputactivations + network["l1biases"])
    thirdlayer = relu(network["l2weights"].T @ secondlayer + network["l2biases"])
    return (network["l3weights"].T @ thirdlayer + network["l3biases"])

def playgame(nn1, nn2, seed):
    rng = np.random.default_rng(seed)
    currgame = pong.new_game(720, 1080, rng.integers(low=0, high=1000))
    actions = ["left", None, "right"]

    for _ in range(6000):
        nn1move = actions[np.argmax(forward_pass(currgame, nn1))]
        nn2move = actions[np.argmax(forward_pass(pong.flipsides(currgame), nn2))]
        currgame["playeraction"] = nn1move
        currgame["opponentaction"] = nn2move
        pong.frame_step(currgame)
        if currgame["state"] == "won":
            return (150*currgame["num_hits"], 50*currgame["num_hits"])
        if currgame["state"] == "lost":
            return (50*currgame["num_hits"], 150*currgame["num_hits"])
    return (90*currgame["num_hits"], 90*currgame["num_hits"])

def playballtracker(nn, seed):
    rng = np.random.default_rng(seed)
    currgame = pong.new_game(720, 1080, rng.integers(low=0, high=1000))
    actions = ["left", None, "right"]

    for _ in range(6000):
        nnmove = actions[np.argmax(forward_pass(currgame, nn))]
        forwardpassmove = actions[np.argmax(ball_tracker(pong.flipsides(currgame)))]
        currgame["playeraction"] = nnmove
        currgame["opponentaction"] = forwardpassmove
        pong.frame_step(currgame)
        if currgame["state"] == "won":
            return 1200*currgame["num_hits"]
        if currgame["state"] == "lost":
            return 600*currgame["num_hits"]
    return 1000*currgame["num_hits"]

def generate(num, seed):
    rng = np.random.default_rng(seed)
    nnseeds = rng.integers(low=0, high=10000, size=num)
    
    output = np.array([new_network(7, 7, 7, 3, nnseed) for nnseed in nnseeds])

    return output

def mutate(nn, rate, seed, scalefactor=0.3):
    newnn = copy.deepcopy(nn)
    rng = np.random.default_rng(seed)
    for row in range(len(nn["l1weights"])):
        for val in range(len(nn["l1weights"][row])):
            if rng.uniform(0, 1) <= rate:
                newnn["l1weights"][row][val] += rng.normal(loc=0, scale=scalefactor, size=None)
            newnn["l1weights"][row][val] = max(newnn["l1weights"][row][val], -5)
            newnn["l1weights"][row][val] = min(newnn["l1weights"][row][val], 5)
    for val in range(len(nn["l1biases"])):
        if rng.uniform(0, 1) <= rate:
            newnn["l1biases"][val] += rng.normal(loc=0, scale=scalefactor, size=None)
        newnn["l1biases"][val] = max(newnn["l1biases"][val], -5)
        newnn["l1biases"][val] = min(newnn["l1biases"][val], 5)
    for row in range(len(nn["l2weights"])):
        for val in range(len(nn["l2weights"][row])):
            if rng.uniform(0, 1) <= rate:
                newnn["l2weights"][row][val] += rng.normal(loc=0, scale=scalefactor, size=None)
            newnn["l2weights"][row][val] = max(newnn["l2weights"][row][val], -5)
            newnn["l2weights"][row][val] = min(newnn["l2weights"][row][val], 5)
    
    for val in range(len(nn["l2biases"])):
        if rng.uniform(0, 1) <= rate:
            newnn["l2biases"][val] += rng.normal(loc=0, scale=scalefactor, size=None)
        newnn["l2biases"][val] = max(newnn["l2biases"][val], -5)
        newnn["l2biases"][val] = min(newnn["l2biases"][val], 5)
    for row in range(len(nn["l3weights"])):
        for val in range(len(nn["l3weights"][row])):
            if rng.uniform(0, 1) <= rate:
                newnn["l3weights"][row][val] += rng.normal(loc=0, scale=scalefactor, size=None)
            newnn["l3weights"][row][val] = max(newnn["l3weights"][row][val], -5)
            newnn["l3weights"][row][val] = min(newnn["l3weights"][row][val], 5)
    
    for val in range(len(nn["l3biases"])):
        if rng.uniform(0, 1) <= rate:
            newnn["l3biases"][val] += rng.normal(loc=0, scale=scalefactor, size=None)
        newnn["l3biases"][val] = max(newnn["l3biases"][val], -5)
        newnn["l3biases"][val] = min(newnn["l3biases"][val], 5) 
    return newnn

def generation_increment(nns, lastscores, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=10000, size=3)
    scores = np.zeros(len(nns))
    for game_seed in seeds:
        for nn1 in range(len(nns)):
            for nn2 in range(len(nns)):
                if nn1 == nn2:
                    continue
                points = playgame(nns[nn1], nns[nn2], game_seed)
                scores[nn1] += points[0]
                scores[nn2] += points[1]

        for nn in range(len(nns)):
            scores[nn] += playballtracker(nns[nn], game_seed)

    for i in range(len(scores)):
        scores[i] = scores[i] + lastscores[i]/2

    nextgen = []
    top3 = np.argpartition(scores, -3)[-3:]
    for k in top3:
        nextgen.append(nns[k])
    #top parents from previous generation continues to live in next generation


    ranks = (np.argsort(np.argsort(-scores)))
    keep = max(1, len(scores) // 4)
    weights = np.maximum(0, keep - ranks)      # best gets `keep`, then keep-1, ..., 0 beyond the top 25%
    prob = weights / weights.sum()
    

    for _ in range(len(nns) - 3 - min(3, int(len(nns)/4))):
        newnn = mutate(rng.choice(nns, p=prob), 0.2, rng.integers(low=0, high=1000))
        nextgen.append(newnn)
    
    nnseeds = rng.integers(low=0, high=10000, size=min(3, int(len(nns)/4)))
        
    for _ in range(min(3, int(len(nns)/4))):
        newnn = new_network(7, 7, 7, 3, nnseeds[_])
        nextgen.append(newnn)
    

    nextscores = np.zeros(len(nns))
    nextscores[:3] = scores[top3]
    return (nextgen, nextscores)


if __name__ == "__main__":
    curr = generate(40, 0)
    rng = np.random.default_rng(12398)
    scores = np.zeros(40)
    for _ in range(10000):
        curr, scores = generation_increment(curr, scores, rng.integers(low=0, high=10000, size=None))

        if _ %10 == 0:
            print(str(_) + " completed generations")
            print(curr[0])
        if _%100 == 0:
            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(str(curr))

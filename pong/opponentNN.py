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

def trackertoNN():
    return {
        "NN" : "tracker"
    }
    

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
            return (150*currgame["player_hits"], 50*currgame["opponent_hits"])
        if currgame["state"] == "lost":
            return (50*currgame["player_hits"], 150*currgame["opponent_hits"])
    return (90*currgame["player_hits"], 90*currgame["opponent_hits"])

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
            return (150*currgame["num_hits"], 50*currgame["num_hits"])
        if currgame["state"] == "lost":
            return (50*currgame["num_hits"], 150*currgame["num_hits"])
    return (90*currgame["num_hits"], 90*currgame["num_hits"])

def trackervtracker(seed):
    rng = np.random.default_rng(seed)
    currgame = pong.new_game(720, 1080, rng.integers(low=0, high=1000))
    actions = ["left", None, "right"]

    for _ in range(6000):
        tracker1move = actions[np.argmax(ball_tracker(currgame))]
        tracker2move = actions[np.argmax(ball_tracker(pong.flipsides(currgame)))]
        currgame["playeraction"] = tracker1move
        currgame["opponentaction"] = tracker2move
        pong.frame_step(currgame)
        if currgame["state"] == "won":
            return (150*currgame["num_hits"], 50*currgame["num_hits"])
        if currgame["state"] == "lost":
            return (50*currgame["num_hits"], 150*currgame["num_hits"])
    return (90*currgame["num_hits"], 90*currgame["num_hits"])


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

def generation_increment(nns, seed, top5, recursionnum=5):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=10000, size=(len(nns)*5))
    scores = np.zeros(len(nns))
    top5scores = np.zeros(5)
    for nn in range(len(nns)):
        for opp in range(5):
            for newseed in range(recursionnum):
                if top5[opp] == trackertoNN():
                    score = playballtracker(nns[nn], seeds[nn*5+opp] + newseed)
                    scores[nn] += score[0]
                else:
                    score = playgame(nns[nn], top5[opp], seeds[nn*5+opp] + newseed)
                    scores[nn] += score[0]
    moreseeds = rng.integers(low=0, high=10000, size=(5, 5, recursionnum))

    for opp1 in range(5):
        for opp2 in range(5):
            for newseed in range(recursionnum):
                if top5[opp1] == trackertoNN() and top5[opp2] == trackertoNN():
                    score = trackervtracker(moreseeds[opp1][opp2][newseed])
                    top5scores[opp1] += score[0]/2
                    top5scores[opp2] += score[1]/2
                elif top5[opp1] == trackertoNN() and top5[opp2] != trackertoNN():
                    score = playballtracker(top5[opp2], moreseeds[opp1][opp2][newseed])
                    top5scores[opp1] += score[1]/2
                    top5scores[opp2] += score[0]/2
                elif top5[opp2] == trackertoNN() and top5[opp1] != trackertoNN():
                    score = playballtracker(top5[opp1], moreseeds[opp1][opp2][newseed])
                    top5scores[opp1] += score[0]/2
                    top5scores[opp2] += score[1]/2
                else:
                    score = playgame(top5[opp1], top5[opp2], moreseeds[opp1][opp2][newseed])
                    top5scores[opp1] += score[0]/2
                    top5scores[opp2] += score[1]/2



    newtop5 = []
    descindices = np.argsort(scores)[::-1]
    desctop5indices = np.argsort(top5scores)[::-1]
    origtop5index = 0
    scoreindex = 0
    while len(newtop5) < 5:
        if top5scores[desctop5indices[origtop5index]] < scores[descindices[scoreindex]]:
            newtop5.append(nns[descindices[scoreindex]])

            scoreindex += 1
        else:
            newtop5.append(top5[desctop5indices[origtop5index]])
            origtop5index += 1

    childseeds = rng.integers(low=0, high=10000, size=len(nns))
    nextgen = []
    for _ in range(len(nns)):
        parent = nns[rng.choice(descindices[:5], p=[.4, .25, .15, .1, .1])]
        nextgen.append(mutate(parent, 0.2, childseeds[_]))

    return (newtop5, nextgen)
    """
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
        scores[i] = scores[i] + lastscores[i]/6
        
    nextgen = []
    top3 = np.argpartition(scores, -3)[-3:]
    for k in top3:
        nextgen.append(nns[k])
    #top parents from previous generation continues to live in next generation


    ranks = (np.argsort(np.argsort(-scores)))
    keep = max(1, len(scores) // 4)
    weights = np.maximum(0, keep - ranks)  
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
"""






if __name__ == "__main__":
    curr = generate(40, 0)
    top5 = np.array([trackertoNN(),trackertoNN(),trackertoNN(),trackertoNN(),trackertoNN()])
    rng = np.random.default_rng(12398)
    scores = np.zeros(40)
    for gen in range(10000):
        top5, curr = generation_increment(curr, rng.integers(low=0, high=10000, size=None), top5)
        if gen %10 == 0:
            print(str(gen) + " completed generations")
            print(curr[0])
        if gen%100 == 0:
            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(str(gen) + " completed generations")
                file.write(str(curr))

"""
Pong game.
"""
import numpy as np

def new_game(width, height, seed):
    rng = np.random.default_rng(seed)
    return {
        "width" : width,
        "height" : height,
        "ballx" : width/2,
        "bally" : height/2,
        "ballspeed" : 5,
        "ballangle" : np.pi* rng.uniform(0.3, 0.7) * rng.choice((-1, 1)),
        "radius" : 10,
        "playerpaddle" : int(width / 2),
        "opponentpaddle" : int(width / 2),
        "paddlespeed" : 3,
        "state" : "ongoing",
        "playeraction" : None,
        "opponentaction" : None
    }


def render(game):
    output = [[' ' for a in range(game["width"])] for b in range(game["height"])]
    for row in output:
        row[0] = "W"
        row[-1] = "W"
    for r in range(int(game["bally"]) - game["radius"], int(game["bally"]) + game["radius"] + 1):
        for c in range(int(game["ballx"]) - game["radius"], int(game["ballx"]) + game["radius"] + 1):
            if 0 <= r < game["height"] and 0 <= c < game["width"]:
                output[r][c] = "B"
    for c in range(game["playerpaddle"] - int(game["width"]/15), game["playerpaddle"] + int(game["width"]/15) + 1):
        for _ in range(game["radius"] + 1):
            output[game["height"] - 30 - _][c] = "P"
    for c in range(game["opponentpaddle"] - int(game["width"]/15), game["opponentpaddle"] + int(game["width"]/15) + 1):
        for _ in range(game["radius"] + 1):
            output[30 - _][c] = "P"
    return output

def frame_step(game):
    if game["playeraction"] == "left":
        game["playerpaddle"] = max(int(game["width"]/15), game["playerpaddle"] - game["paddlespeed"])
    if game["playeraction"] == "right":
        game["playerpaddle"] = min(game["width"] - int(game["width"]/15) - 1, game["playerpaddle"] + game["paddlespeed"])
    if game["opponentaction"] == "left":
        game["opponentpaddle"] = max(int(game["width"]/15), game["opponentpaddle"] - game["paddlespeed"])
    if game["opponentaction"] == "right":
        game["opponentpaddle"] = min(game["width"] - int(game["width"]/15) - 1, game["opponentpaddle"] + game["paddlespeed"])

    newballx = game["ballx"] + game["ballspeed"] * np.cos(game["ballangle"])
    newbally = game["bally"] + game["ballspeed"] * np.sin(game["ballangle"])

    playerpaddle = [game["playerpaddle"] - int(game["width"]/15), 
                    game["height"] - 30 - game["radius"],
                    game["playerpaddle"] + int(game["width"]/15), 
                    game["height"] - 30]
    opponentpaddle = [game["opponentpaddle"] - int(game["width"]/15), 
                    30 - game["radius"],
                    game["opponentpaddle"] + int(game["width"]/15), 
                    30]

    newball = [newballx - game["radius"], newbally - game["radius"], 
               newballx + game["radius"], newbally + game["radius"]]

    playerballintersect = [max(playerpaddle[0], newball[0]), max(playerpaddle[1], newball[1]),
                           min(playerpaddle[2], newball[2]), min(playerpaddle[3], newball[3])]
    if playerballintersect[2] >= playerballintersect[0] and playerballintersect[3] >= playerballintersect[1]:
        newbally -= playerballintersect[3] - playerballintersect[1]
        offset = np.clip((newballx - game["playerpaddle"])/int(game["width"]/15), -1, 1)
        game["ballangle"] = -(np.pi/2 + offset * np.pi/3)
    
    opponentballintersect = [max(opponentpaddle[0], newball[0]), max(opponentpaddle[1], newball[1]),
                           min(opponentpaddle[2], newball[2]), min(opponentpaddle[3], newball[3])]
    
    if opponentballintersect[2] >= opponentballintersect[0] and opponentballintersect[3] >= opponentballintersect[1]:
        newbally += opponentballintersect[3] - opponentballintersect[1]
        offset = np.clip((newballx - game["opponentpaddle"])/int(game["width"]/15), -1, 1)
        game["ballangle"] = np.pi/2 + offset * np.pi/3

    if newballx < game["radius"] + 1:
        newballx = game["radius"] + 1
        if np.cos(game["ballangle"]) <= 0:
            game["ballangle"] = np.pi - game["ballangle"]
    if newballx > game["width"] - game["radius"] - 1:
        newballx = game["width"] - game["radius"] - 1
        if np.cos(game["ballangle"]) >= 0:
            game["ballangle"] = np.pi - game["ballangle"]

    game["ballx"] = newballx
    game["bally"] = newbally

    if game["bally"] > game["height"] - 20:
        game["state"] = "lost"
    if game["bally"] < 20:
        game["state"] = "won"

def flipsides(game):
    output = {
        "width" : game["width"],
        "height" : game["height"],
        "ballx" : game["ballx"],
        "bally" : game["height"] - game["bally"],
        "ballspeed" : game["ballspeed"],
        "ballangle" : -game["ballangle"],
        "radius" : game["radius"],
        "playerpaddle" : game["opponentpaddle"],
        "opponentpaddle" : game["playerpaddle"],
        "paddlespeed" : game["paddlespeed"],
        "playeraction" : game["opponentaction"],
        "opponentaction" : game["playeraction"]
    }
    if game["state"] == "won":
        output["state"] = "lost"
    elif game["state"] == "lost":
        output["state"] = "won"
    else:
        output["state"] = "ongoing"
    return output

if __name__ == "__main__":
    game = new_game(100, 100, 0)
    print(game)
    print(flipsides(game))
    

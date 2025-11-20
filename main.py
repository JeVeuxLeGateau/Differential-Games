"""
Main function for playing games
Here is where you choose each game you want to play, set the parameters,
and handle the logic of what policy controls which player

Implemented games
Homicidal_Chauffeur -> Homicidal_Chauffeur.py
"""
import time
import Games.Homicidal_Chauffeur as hc
import numpy as np
import policies as pol


def main():
    chauffeur_omega = 1
    chauffeur_speed = 10
    victim_speed = 0.0

    timer = 1
    time_limit = 10000
    dt = 0.1
    timesteps = int(time_limit/dt)
    box_dimensions = [1000, 1000]
    status = 0
    game = hc.Homicidal_Chauffeur(chauffeur_omega, chauffeur_speed, victim_speed, box_dimensions, dt)


    actions = {"chauffeur": ,
               "victim": np.array(0)}

    while timer <= timesteps and status == 0:
        game.step(actions)
        status = game.check_win()
        game.render()
        time.sleep(0.001)
        timer += 1

    if timer == timesteps:
        print("WIN!!!!!!")
    else:
        print("Wah Waaaaah you lose...")

    print("Lasted for", round(timer*dt,2), "seconds!")



main()
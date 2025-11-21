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
# import policies.pure_pursuit as pp
# import policies.zig_zag as zz
import policies as p


def main():
    chauffeur_omega = 1
    chauffeur_speed = 20
    victim_speed = 10

    timer = 1
    time_limit = 10000
    dt = 0.1
    timesteps = int(time_limit / dt)
    box_dimensions = [1000, 1000]
    status = 0
    game = hc.Homicidal_Chauffeur(chauffeur_omega, chauffeur_speed, victim_speed, box_dimensions, dt)

    chauffeur = game.chauffeur
    victim = game.victim

    while timer <= timesteps and status == 0:
        c_dict = chauffeur.get_state_dict()
        v_dict = victim.get_state_dict()

        c_cmd = p.pure_pursuit(c_dict, v_dict)
        v_cmd = p.random_motion(T=200, current_time=timer, cur_theta=victim.theta)

        actions = {"chauffeur": c_cmd,
                   "victim": v_cmd}

        game.step(actions)
        game.enforce_bounds()

        status = game.check_win()
        game.render()
        time.sleep(0.001)
        timer += 1

    if timer == timesteps:
        print("WIN!!!!!!")
    else:
        print("Wah Waaaaah you lose...")

    print("Lasted for", round(timer * dt, 2), "seconds!")


main()

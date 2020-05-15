"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
import numpy as np
from mlgame.communication import ml as comm

def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    blocker_last_x = 0

    class Pred:
        pred = 100
        blocker_pred_x = 0
        last_command = 0
        blocker_vx = 0

    
    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left
        else :
            if scene_info["platform_2P"][0]+20  > (pred-10) and scene_info["platform_2P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_2P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left

    def ml_loop_for_1P(): 
        # ball slicing
        if scene_info["ball_speed"][1] > 0 and (scene_info["ball"][1]+scene_info["ball_speed"][1]) >= 415 and Pred.last_command == 0:
            print("------")
            ball_x = scene_info["ball"][0]
            ball_y = scene_info["ball"][1]
            ball_vx = scene_info["ball_speed"][0]
            ball_slice_vx = scene_info["ball_speed"][0]+np.sign(scene_info["ball_speed"][0])*3
            ball_vy = scene_info["ball_speed"][1] 
            blocker_x = scene_info['blocker'][0] + Pred.blocker_vx
            
            y = abs((415 - ball_y) // ball_vy)
            pred_ball_1P = ball_x + ball_vx * y

            y = abs((415 - 260) // ball_vy)
            pred_ball_blocker = pred_ball_1P + ball_slice_vx * y
            bound = pred_ball_blocker // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    pred_ball_blocker = pred_ball_blocker - bound*200                    
                else :
                    pred_ball_blocker = 200 - (pred_ball_blocker - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    pred_ball_blocker = abs(pred_ball_blocker - (bound+1) *200)
                else :
                    pred_ball_blocker = pred_ball_blocker + (abs(bound)*200)
            
            y = abs((415 - 260) // ball_vy)
            Pred.blocker_pred_x = blocker_x + Pred.blocker_vx * y 
            if Pred.blocker_pred_x < 0: Pred.blocker_pred_x = abs(Pred.blocker_pred_x)
            elif Pred.blocker_pred_x > 170: Pred.blocker_pred_x = 170 - (Pred.blocker_pred_x - 170)
            
            if pred_ball_blocker >= Pred.blocker_pred_x-10 and pred_ball_blocker < Pred.blocker_pred_x+40:
                print("slice will hit blicker")
                # don't slice 
                # use origin ball vx to predict will hit blocker or not
                # if will hit blicker let ball go reverse direction
                y = abs((415 - 260) // ball_vy)
                pred_ball_blocker = pred_ball_1P + ball_vx * y
                bound = pred_ball_blocker // 200 # Determine if it is beyond the boundary
                if (bound > 0): # pred > 200 # fix landing position
                    if (bound%2 == 0) : 
                        pred_ball_blocker = pred_ball_blocker - bound*200                    
                    else :
                        pred_ball_blocker = 200 - (pred_ball_blocker - 200*bound)
                elif (bound < 0) : # pred < 0
                    if (bound%2 ==1) :
                        pred_ball_blocker = abs(pred_ball_blocker - (bound+1) *200)
                    else :
                        pred_ball_blocker = pred_ball_blocker + (abs(bound)*200)

                if pred_ball_blocker >= Pred.blocker_pred_x-10 and pred_ball_blocker < Pred.blocker_pred_x+40:
                    print("will hit blocker, hit reversed direction")
                    if scene_info["ball_speed"][0] > 0: return 2
                    else: return 1
                else: 
                    print("will not hit blicker, do nothing")
                    return 0
            else:
                # slice
                print("slice will not hit blocker")
                if scene_info["ball_speed"][0] > 0: return 1
                else: return 2

        elif scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            Pred.pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
            bound = Pred.pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    Pred.pred = Pred.pred - bound*200                    
                else :
                    Pred.pred = 200 - (Pred.pred - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    Pred.pred = abs(Pred.pred - (bound+1) *200)
                else :
                    Pred.pred = Pred.pred + (abs(bound)*200)
            return move_to(player = '1P',pred = Pred.pred)
                
        else : # 球正在向上 # ball goes up
            return move_to(player = '1P',pred = 100)



    def ml_loop_for_2P():  # as same as 1P
        if scene_info["ball_speed"][1] > 0 : 
            return move_to(player = '2P',pred = 100)
        else : 
            x = ( scene_info["platform_2P"][1]+30-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x) 
            bound = pred // 200 
            if (bound > 0):
                if (bound%2 == 0):
                    pred = pred - bound*200 
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) :
                if bound%2 ==1:
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            return move_to(player = '2P',pred = pred)

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            blocker_last_x = scene_info["blocker"][0]
            Pred.last_command = 0
            ball_served = True
        else:
            if side == "1P":
                Pred.blocker_vx = scene_info["blocker"][0] - blocker_last_x
                if scene_info["blocker"][0] == 0: Pred.blocker_vx = 5
                elif scene_info["blocker"][0] == 170: Pred.blocker_vx = -5
                command = ml_loop_for_1P()
                blocker_last_x = scene_info["blocker"][0]
                Pred.last_command = command
            else:
                command = ml_loop_for_2P()

            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
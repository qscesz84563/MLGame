"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
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
    blockervx = 0

    def check_hit_blocker(ballx, bally, ballvx, ballvy, blocker_pred_x):
        y = abs((bally - 260) // ballvy)
        pred = ballx + ballvx * y
        if pred < 0:
            x = abs(ballx // ballvx)
            bally = bally + ballvy * x
            ballx = 0
            ballvx *= -1
            y = abs((bally - 260) // ballvy)
            pred = ballx + ballvx * y
        else:
            x = abs((ballx - 200)// ballvx)
            bally = bally + ballvy * x
            ballx = 200
            ballvx *= -1
            y = abs((bally - 260) // ballvy)
            pred = ballx + ballvx * y
        
        if(pred >= blocker_pred_x and pred <= blocker_pred_x+30):
            ballx = pred
            bally = 260
            ballvx *= -1
            ballvy *= -1
            y = abs((420 - 260) // ballvy)
            pred = ballx + ballvx * y
            if pred < 0:
                x = abs(ballx // ballvx)
                bally = bally + ballvy * x
                ballx = 0
                ballvx *= -1
                y = abs((bally - 420) // ballvy)
                pred = ballx + ballvx * y
            else:
                x = abs((ballx - 200)// ballvx)
                bally = bally + ballvy * x
                ballx = 200
                ballvx *= -1
                y = abs((bally - 420) // ballvy)
                pred = ballx + ballvx * y
        else: pred = 100

        return pred


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
        if(scene_info["ball"][1] >= 240 and scene_info["ball_speed"][1] < 0):
            ballx = scene_info["ball"][0]
            bally = scene_info["ball"][1]
            ballvx = scene_info["ball_speed"][0]
            ballvy = scene_info["ball_speed"][1]
            blocker_pred_x = scene_info["blocker"][0] + blockervx*(abs((bally - 260) // ballvy))
            if blocker_pred_x < 0: blocker_pred_x = abs(blocker_pred_x)
            elif blocker_pred_x > 170: blocker_pred_x = 170 - (abs(blocker_pred_x) - 170)

            pred = check_hit_blocker(ballx=ballx, bally=bally, ballvx=ballvx, ballvy=ballvy, blocker_pred_x=blocker_pred_x)
            return move_to(player = '1P',pred = pred)
        
        elif scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
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
            return move_to(player = '1P',pred = pred)

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
            ball_served = True
        else:
            if side == "1P":
                blockervx = scene_info["blocker"][0] - blocker_last_x
                blocker_last_x = scene_info["blocker"][0]
                command = ml_loop_for_1P()
            else:
                command = ml_loop_for_2P()

            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
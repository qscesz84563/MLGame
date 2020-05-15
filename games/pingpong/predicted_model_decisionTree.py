import pickle
import numpy as np
import os
from os import path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Pred:
    pred = 100
    blocker_pred_x = 0
    last_command = 0
    blocker_vx = 0


def transformCommand(command):
    if 'MOVE_RIGHT' == command:
        return 1
    elif 'MOVE_LEFT' == command:
        return 2
    else:
        return 0

if __name__ == '__main__':
    # read all file under log   
    # 因為 pickle 存的是放在列表裡的字典，先宣告一個空列表
    Data = list()
    # 目標檔案夾
    folder = path.join(path.dirname(__file__), 'log')
    # 檔案夾下的所有檔案名稱  
    files = os.listdir(folder)
    for f in files:
        # 絕對路徑
        fullpath = path.join(folder, f)
        with open(fullpath, 'rb') as file:
            # handle 為放字典的列表
            handle = pickle.load(file)
            # 將字典取出，再放入 data 列表
            for i in range(len(handle)):
                Data.append(handle[i])
    data = np.zeros((len(Data), 1))
    # 1
    Ball_x = []
    for i in range(len(Data)):
        Ball_x.append(Data[i]['ball'][0])
    Ball_x = np.array(Ball_x)
    Ball_x = Ball_x.reshape(len(Ball_x), 1)
    data = np.hstack((data, Ball_x))
    # 2
    Ball_y = []
    for i in range(len(Data)):
        Ball_y.append(Data[i]['ball'][1])
    Ball_y = np.array(Ball_y)
    Ball_y = Ball_y.reshape(len(Ball_y), 1)
    data = np.hstack((data, Ball_y))
    # 3
    Ball_vx = []
    for i in range(len(Data)):
        Ball_vx.append(Data[i]['ball_speed'][0])
    Ball_vx = np.array(Ball_vx)
    Ball_vx = Ball_vx.reshape(len(Ball_vx), 1)
    data = np.hstack((data, Ball_vx))
    # 4
    Ball_vy = []
    for i in range(len(Data)):
        Ball_vy.append(Data[i]['ball_speed'][1])
    Ball_vy = np.array(Ball_vy)
    Ball_vy = Ball_vy.reshape(len(Ball_vy), 1)
    data = np.hstack((data, Ball_vy))
    # 5
    plat1_x = []
    for i in range (len(Data)):
        plat1_x.append(Data[i]['platform_1P'][0])
    plat1_x = np.array(plat1_x)
    plat1_x = plat1_x.reshape(len(plat1_x), 1)
    data = np.hstack((data, plat1_x))
    # 6
    plat2_hit_ball_x = []
    for i in range(len(Data)):
        if Data[i]["ball"][1] == 80:
            plat2_hit_ball_x.append(Data[i]["ball"][0])
        else: plat2_hit_ball_x.append(1000)
    plat2_hit_ball_x = np.array(plat2_hit_ball_x)
    plat2_hit_ball_x = plat2_hit_ball_x.reshape(len(plat2_hit_ball_x), 1)
    data = np.hstack((data, plat2_hit_ball_x))
    # 7
    blocker_x = []
    for i in range(len(Data)):
        blocker_x.append(Data[i]['blocker'][0])
    blocker_x = np.array(blocker_x)
    blocker_x = blocker_x.reshape(len(blocker_x), 1)
    data = np.hstack((data, blocker_x))
    # 8
    pred_origin = []
    # 9
    pred_minus5 = []
    # 10
    pred_minus10 = []
    # 11
    pred_minus15 = []
    # 12
    pred_add5 = []
    # 13
    pred_add10 = []
    # 14
    pred_add15 = []

    for i in range(len(Data)):
        if i == 0:
            Pred.last_command = 0
        else:
            Pred.last_command = Data[i - 1]["command_1P"]

        if i == len(Data) - 1:
            blocker_vx = Data[i]["blocker"][0] - Data[i - 1]["blocker"][0]
        else:
            blocker_vx = Data[i + 1]["blocker"][0] - Data[i]["blocker"][0]

        if Data[i]["ball_speed"][1] > 0 and (Data[i]["ball"][1]+Data[i]["ball_speed"][1]) >= 415 and Pred.last_command == 0:
            y = abs((415 - Data[i]["ball"][1]) // Data[i]["ball_speed"][1])
            Pred.pred = Data[i]["ball"][0] + Data[i]["ball_speed"][0] * y

        elif Data[i]["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( Data[i]["platform_1P"][1]-Data[i]["ball"][1] ) // Data[i]["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            Pred.pred = Data[i]["ball"][0]+(Data[i]["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
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
        else : # 球正在向上 # ball goes up
            Pred.pred = 100

        pred_origin.append(Pred.pred)
        pred_minus5.append(Pred.pred - 5)
        pred_minus10.append(Pred.pred - 10)
        pred_minus15.append(Pred.pred - 15)
        pred_add5.append(Pred.pred + 5)
        pred_add10.append(Pred.pred + 10)
        pred_add15.append(Pred.pred + 15)
    
    pred_origin = np.array(pred_origin)
    pred_origin = pred_origin.reshape(len(pred_origin), 1)
    data = np.hstack((data, pred_origin))

    pred_minus5 = np.array(pred_minus5)
    pred_minus5 = pred_minus5.reshape(len(pred_minus5), 1)
    data = np.hstack((data, pred_minus5))

    pred_minus10 = np.array(pred_minus10)
    pred_minus10 = pred_minus10.reshape(len(pred_minus10), 1)
    data = np.hstack((data, pred_minus10))

    pred_minus15 = np.array(pred_minus15)
    pred_minus15 = pred_minus15.reshape(len(pred_minus15), 1)
    data = np.hstack((data, pred_minus15))

    pred_add5 = np.array(pred_add5)
    pred_add5 = pred_origin.reshape(len(pred_add5), 1)
    data = np.hstack((data, pred_add5))

    pred_add10 = np.array(pred_add10)
    pred_add10 = pred_add10.reshape(len(pred_add10), 1)
    data = np.hstack((data, pred_add10))

    pred_add15 = np.array(pred_add15)
    pred_add15 = pred_add15.reshape(len(pred_add15), 1)
    data = np.hstack((data, pred_add15))

    # 15
    command_1P = []
    for i in range(len(Data)):
        command_1P.append(transformCommand(Data[i]['command_1P']))
    command_1P = np.array(command_1P)
    command_1P = command_1P.reshape(len(command_1P), 1)
    data = np.hstack((data, command_1P))


    data = data[:,1:]

    X = data[:, :-1]
    Y = data[:, -1]

    x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)     
    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    print(y_predict)
    count0 = count1 = count2 = 0
    for i in range(len(y_predict)):
        if(y_predict[i] == 0):
            count0 += 1
        elif(y_predict[i] == 1):
            count1 += 1
        else:
            count2 += 1
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))

    print(count0)
    print(count1)
    print(count2)

    with open('games/pingpong/ml/save/trained_model.pickle', 'wb') as f:
        pickle.dump(tree, f)


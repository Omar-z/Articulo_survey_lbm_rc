import numpy as np
import torch
import matplotlib.pyplot as plt

def __R__(X,YH,T):
    tm = T.mean()
    yhm = YH.mean()
    a = ((T-tm)*(YH-yhm)).sum()
    b1 = ((T-tm)**2).sum()
    b2 = ((YH-yhm)**2).sum()
    return a.cpu().detach().numpy()/np.sqrt(b1.cpu().detach().numpy()*b2.cpu().detach().numpy())

def __R2__(YH,T):
    tm = T.mean()
    a = ((T-YH)**2).sum()
    b = ((T-tm)**2).sum()
    return 1-(a.cpu().detach().numpy()/b.cpu().detach().numpy())

def confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int,plot=True,debug=True) -> torch.Tensor:
    pred_labels = y_pred.argmax(dim=1).long()
    y_true = y_true.argmax(dim=1).long()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, pred_labels):
        cm[t, p] += 1
    
    if debug:
        print(cm)
        print("Accuracy: {0:.3f}\nPrecision: {1:.3f}".format(*get_accuracy_precision(cm)))  
    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('matrix de confusion')
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j].item()
                plt.text(j, i, str(count),
                        ha='center', va='center',
                        color='white' if count > cm.max()/2 else 'black')
        plt.show()
    return get_accuracy_precision(cm)

def get_accuracy_precision(cm: torch.Tensor):
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    accuracy = correct / total if total else 0

    num_classes = cm.size(0)
    precisions = []
    for i in range(num_classes):
        col_sum = cm[:, i].sum().item()
        if col_sum == 0:
            precisions.append(0.0)
        else:
            precisions.append(cm[i, i].item() / col_sum)
    precision_macro = sum(precisions) / num_classes

    return accuracy, precision_macro 

def PlotTraining(X,YH,T,debug=True)->None:
        x = X.cpu().detach().numpy()
        yh = YH.cpu().detach().numpy()
        t = T.cpu().detach().numpy()
        n,_ = x.shape
        _,m = t.shape if len(t.shape) > 1 else (t.shape[0],1)
        B = (x.T@x)**-1@(x.T@t)
        rl = x@B#A*X+B
        SSe = ((t-yh)**2).flatten().sum()
        R = __R__(X,YH,T)
        R2 =__R2__(YH,T)
        if debug:
            print(f"SSE = {SSe:E}")
            print(f"MSE = {(1/n)*SSe:E}")
            print(f"R = {R}")
            print(f"R^2 = {R2}")
        fig,ax = plt.subplots(m,2,figsize=(12,8),sharex=True,sharey=True)

        for ren in range(m):
            if m>1:
                ax[ren ,0].plot(t[:,ren],"-",color='black')
                ax[ren, 0].set_title(f"Target_{ren+1}")
                ax[ren ,1].plot(yh[:,ren],"-",color='blue')
                ax[ren, 1].set_title(f"R={R}\n$R^2$={R2}")
            else:
                ax[0].plot(t[:],"-",color='black')
                ax[0].set_title(f"Target_{ren+1}")
                ax[1].plot(yh[:],"-",color='blue')
                ax[1].set_title(f"R={R}\n$R^2$={R2}")
        fig.tight_layout()
        plt.show()
        return SSe, R, R2

def OneHotEncode(X,clases=2)->torch.Tensor:
    #print(f"X shape: {X.shape},{clases}")
    return torch.nn.functional.one_hot(X.to(torch.long),clases).squeeze().float()

def getLane(OM: np.ndarray,normalizado:bool = False) -> tuple[float,float,float]:
    """
    Función que determina a que distancia se encuentra los carros en cada carril con respecto
    a el agente.\n

    El agente se puede encontrar en cualquiera de los 3 carriles\n 
    (0, arriba), (4, medio), (8, abajo)\n

    y regresa las distancias de los carros relativo al agente\n
    
    agente carril = 0 (medio) en datos 4
    ------------------------
    carril de otros carros
    0  -> medio
    -4 -> arriba
    4  -> abajo

    agente carril = 1 (arriba) en datos 0
    ------------------------
    carril de otros carros
    4  -> medio
    0 -> arriba
    8  -> abajo

    agente carril = -1 (abajo) en datos 8 -> 0.08
    ------------------------
    carril de otros carros
    -4  -> medio : -0.04
    -8 -> arriba.: -0.08
    0  -> abajo. : 0.0


    """
    _, agente_carril_str = QueCarrilVoy(OM[0:1,:],norm=normalizado)
    r, _ = OM.shape
    d_norm = 1 if not normalizado else 100
    sin_valor = -100
    arriba,medio,abajo=sin_valor,sin_valor,sin_valor
    #la matriz esta acomodada del más cerca al mas lejano por lo que al encontrar un carril no se ocupa buscar el otro
    if agente_carril_str == "arriba":
        for i in range(1,r):
            carro = OM[i, 1]
            if (carro >= -2/d_norm and carro <2/d_norm) and arriba == sin_valor:
                arriba = OM[i, 0]
            elif (carro >= 2/d_norm and carro <6/d_norm) and medio == sin_valor:
                medio = OM[i, 0]
            elif (carro>=6/d_norm and carro <= 8/d_norm) and abajo == sin_valor:
                abajo = OM[i, 0]
    elif agente_carril_str == "medio":
        for i in range(1,r):
            carro = OM[i, 1]
            if (carro >= -5/d_norm  and carro<-3/d_norm) and arriba == sin_valor:
                arriba = OM[i, 0]
            elif (carro >= -3/d_norm and carro <2/d_norm) and medio == sin_valor:
                medio = OM[i, 0]
            elif (carro >= 2/d_norm and carro <=5/d_norm) and abajo == sin_valor:
                abajo = OM[i, 0]
    elif agente_carril_str == "abajo":
        for i in range(1,r):
            carro = OM[i, 1]
            if (carro >= -8/d_norm and carro <=-6/d_norm) and arriba == sin_valor:
                arriba = OM[i, 0]
            elif (carro > 6/d_norm and carro<= -2/d_norm) and medio == sin_valor:
                medio = OM[i, 0]
            elif (carro > -2/d_norm  and carro <=2/d_norm) and abajo == sin_valor:
                abajo = OM[i, 0]  
    
    return arriba, abajo, medio
    

def GetActionNumber(raw_num, version=2):
    if(version==2): # es salidas
        if raw_num >2:
            return 1
        return raw_num
    # 5 salidas
    return raw_num

def QueCarrilVoy(obs: np.ndarray,ant =None,norm:bool=False) -> tuple[int,str]:
    """
    Determina en que carril se encuentra el agente\n
    -1: abajo\n
    0: medio\n
    1: arriba\n

    """
    n_div = 1 if not norm else 100
    if obs[0,1] >= -7/n_div and obs[0,1] < 3/n_div:
        return 1,"arriba"
    elif obs[0,1] >=7/n_div and obs[0,1]<= 20/n_div:
        return -1,"abajo"
    elif obs[0,1] >=3/n_div and obs[0,1]< 7/n_div:
        return 0,"medio"
    return ant,"no se sabe"

def actionToStr(actions: int) -> str:
    """
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
    """
    if actions == 0:
        return "LANE_LEFT"
    elif actions == 1:
        return "IDLE"
    elif actions == 2:
        return "LANE_RIGHT"
    elif actions == 3:
        return "FASTER"

    return "SLOWER"

def RetroAlimentacionBaseReglas(estado:list[int,float,float,float],estado_velocidad:int, **args)->int:
    """
    Función que le dice al modelo cual era la acción mas probable de hacer en una instancia determinada
    el peso de las acciones en este caso es lo mismo (1) con excepción de la acción velocidad
    params
    @estado: lista de 4 elementos que representan el estado actual deel mundo (carril_agente, carril_arriba, carril_medio, carril_abajo)
    @estado_velocidad: entero que representa la velocidad del agente(-1,0,1)[lento,normal,rapido]
    @reglas: lista por cada situación de reglas que se deben seguir para tomar una acción
    return  int: la acción que se debió haber tomado
    """
    """
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
    """
    #forzar a moverse de carril
    agente = estado[0]
    if(agente == 0): #medio
        #arriba o abajo sin carros
        a_arriba = estado[1]
        a_abajo = estado[3]
        if(a_arriba <30 and a_abajo >30):
            opciones=[2]
            return np.random.choice(opciones)
        elif(a_arriba >30 and a_abajo <30):
            opciones=[0]
            return np.random.choice(opciones)
        elif(a_arriba >30 and a_abajo >30):
            opciones=[0,2]
            return np.random.choice(opciones)
        else:
            return np.random.choice([0,2]);
    elif(agente == 1): #arriba
        opciones = [2]
        return np.random.choice(opciones)

    elif(agente == -1): #abajo
        opciones = [0]
        return np.random.choice(opciones)

    return 1

def PrintLogLevel(level:str, message:str)->None:
    if level not in ["INFO","WARNING","ERROR"]:
        level = "INFO"
    color = "\033[94m" if level == "INFO" else "\033[93m" if level == "WARNING" else "\033[91m"
    print(f"{color}[{level}] {message}\033[0m")
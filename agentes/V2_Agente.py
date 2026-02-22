from utils.Anfis_utils import CargarFIS, CrearFISInicial, GuardarFIS
from anfislm.V2_Anfis import RLANFISBuilder, LevenberMaquardtOpt, train_nfs
#from utils.funciones_auxiliares import OneHotEncode, PlotTraining, confusion_matrix
import torch
import fuzzylab as fz
import seaborn as sns
from utils.funciones_auxiliares import getLane
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import random
import numpy as np
import json
import time
import pickle as pkl
import sys
import os


def loss_cosine_similarity(a:torch.Tensor, b:torch.Tensor,reduction='None')->torch.Tensor:
    
    similitud = torch.nn.CosineSimilarity()(a,b)
    return 1-(similitud.sum() / similitud.size(0)) 

class Memoria(object):
    def __init__(self, max_size:int=10000,agente_config:dict=None) -> None:
        self.capacidad = max_size
        self.a_config = agente_config
        # memoria circular
        self.memoria = []
        self.curr = 0
    
    def guardar_memoria(self, archivo:str) -> None:
        """
        Guarda el objeto de Memoria del agente en un archivo pickle.
        Args:
            archivo (str): Nombre del archivo donde se guardará la memoria (incluir extensión .pkl)
        """
        try:
            with open(archivo, 'wb') as f:
                pkl.dump(self, f)
            print(f"Memoria guardada exitosamente en {archivo}")
        except Exception as e:
            print(f"Error al guardar la memoria: {e}")

    def cargar_memoria(self, archivo:str) -> 'Memoria':
        """
        Carga el objeto de Memoria del agente desde un archivo pickle.
        Args:
            archivo (str): Nombre del archivo desde donde se cargará la memoria (incluir extensión .pkl)
        Returns:
            Memoria: El objeto de Memoria cargado
        """
        try:
            with open(archivo, 'rb') as f:
                memoria_cargada = pkl.load(f)
            print(f"Memoria cargada exitosamente desde {archivo}")
            return memoria_cargada
        except FileNotFoundError:
            print(f"Archivo {archivo} no encontrado")
            return None
        except Exception as e:
            print(f"Error al cargar la memoria: {e}")
            return None
    
    def muestra_batch(self,batch_size:int,continuos:bool=False):
        if(len(self.memoria) < batch_size):
            return None
        tranciciones = self.get_muestra(batch_size,continuos)
        return list(tranciciones)
    
    def get_muestra(self,batch_size:int, continuas:bool=False):
        if(self.a_config["step"]==1):
            return random.sample(self.memoria, batch_size)
        else:
            if not continuas:
                # un lote compuesto de muestras al azar
                indices = random.sample(range(len(self.memoria)), batch_size)
                transiciones = [self.memoria[i:i+self.a_config["step"]] for i in indices]
                #calcular el valor de ∑T(s,a,s',a')
                return map(self.sumTdeSASdAd,transiciones)
            # un lote compuesto con muestras continuas
            id_inicio = random.sample(range(len(self.memoria)-batch_size),1)
            transiciones = [self.memoria[i:i+batch_size] for i in id_inicio]
            return map(self.sumTdeSASdAd,transiciones)
        
    
    def push(self,*args): #args = (estado,accion,recompensa,estado_siguiente,fin,info)
        # mas rápido que alocar memoria al inicio del tamaño
        if(len(self.memoria) < self.capacidad):
            self.memoria.append(None)
            self.curr = len(self.memoria) - 1
            #mas rápido que hacer pop
        elif (len(self.memoria) > self.capacidad):
            self.memoria = self.memoria[:self.capacidad]
        self.memoria[self.curr] = args
        self.curr = (self.curr + 1) % self.capacidad
    
    def __penality__(self,accion,carril)->float:
        if (carril == 1 and accion == 0) or (carril == -1 and accion == 2):
            return 1e-6
        return 1.0

    def sumTdeSASdAd(self,transiciones:list[list]) ->tuple:
        #∑T = T + gamma * T(s,a,s',a') 
        gamma = self.a_config["gamma"]
        descuento= self.a_config["descuento"]
        estado,accion, acarril,ave,sR, estado_siguiente, fin = transiciones[0]
        for T in transiciones[1:]:
            if fin:
                break
            else:
                _,taccion, tcaril,tve,recompensa, estado_siguiente, fin = T
                descuento *= gamma 
                sR += descuento * recompensa * self.__penality__(taccion,acarril)
        return estado,accion,acarril,ave,sR,estado_siguiente,fin
    
    def __len__(self):
        return len(self.memoria)

####prueba unitarias de clase memoria
def test_memoria():
    print("--- Iniciando pruebas para la clase Memoria ---")
    # --- Parámetros de la prueba ---
    capacidad_memoria = 100
    batch_size = 10
    dimension_estado = 4
    dimension_accion = 4
    numero_de_muestras_a_insertar = 25
    config = {"step": 1, "gamma": 0.99, "descuento": 1.0}

    # 1. Crear una instancia de la Memoria
    memoria = Memoria(capacidad_memoria,config)
    print(f"Memoria creada con capacidad: {memoria.capacidad}")

    # 2. Probar la función push
    print(f"\nInsertando {numero_de_muestras_a_insertar} muestras en la memoria...")
    for i in range(numero_de_muestras_a_insertar):
        estado = torch.randn(dimension_estado)
        accion = torch.tensor(float(random.randint(0,dimension_accion)))
        recompensa = torch.tensor(float(i))
        siguiente_estado = torch.randn(dimension_estado)
        hecho = torch.tensor(float(i % 5 == 0)) # Simular que 'hecho' es True cada 5 pasos
        
        memoria.push(estado, accion, recompensa, siguiente_estado, hecho)
    # Verificar que el tamaño de la memoria sea correcto
    assert len(memoria) == numero_de_muestras_a_insertar, \
    f"Error en push: El tamaño de la memoria debería ser {numero_de_muestras_a_insertar}, pero es {len(memoria)}"
    print(f"Prueba de push exitosa. Tamaño actual de la memoria: {len(memoria)}")
    
    # 3. Probar la función get_muestra
    print(f"\nObteniendo una muestra de tamaño {batch_size}...")
    
    e = memoria.muestra_batch(batch_size)
    for i in range(len(e)):
        print(f"{i+1} {e[i]}")
    print(f"Prueba de push exitosa. Muestras mostradas: {batch_size}")
    
    config["step"] = 30
    print(f"\nObteniendo una muestra de tamaño {batch_size} y step {config['step']}...")
    e = memoria.muestra_batch(batch_size)
    for i in range(len(e)):
        print(f"{i+1} {e[i]}")
    
    print(f"Prueba de push exitosa. Muestras mostradas: {batch_size}")
 
#########

class Agente:
    def __init__(self,fis_archivo:str,batch_s:int=32,w_pre=False,memoria_size=10_000,device="cpu",**modelo_args)->None:
        
        self.config = {}
        self.config["step"]=1
        self.config["gamma"]=0.99
        self.config["descuento"]=1.0
        self.config["batch_size"] = batch_s
        self.config["memoria_size"] = memoria_size
        self.config["world_data"] = w_pre # preprocesar los datos del entorno o no para la red criticona
        self.m_fis,self.m_in,self.m_out,self.m_rules = self.__getInfoFis__(fis_archivo)
        # referencia a las variables que se van a observar
        # como el carril que el agente está actualmente
        self.a_carril = -1 
        self.a_velocidad = 0
        lista_argumentos = [i[1] for i in modelo_args.items()]
        self.memoria = Memoria(memoria_size,self.config)

        self.device= device
        print(f"Agente configurado para ejecutarse en: {self.device}")
        if self.device == "cuda":
            print(f"GPU detectada: {torch.cuda.get_device_name()}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif self.device == "mps":
            print("Usando Metal Performance Shaders (MPS) para GPU en macOS.")

        self.model =RLANFISBuilder() \
            .AddFIS(fis_archivo) \
            .AddInputs(self.m_in) \
            .AddOutputs(self.m_out) \
            .AddRules(self.m_rules) \
            .AddMuStats(*lista_argumentos) \
            .AddTipoProblema("clasificacion") \
            .AddDevice(self.device) \
            .AddOptimizador(LevenberMaquardtOpt) \
            .Build()
        
        critic_input = self.m_in
        if self.config["world_data"]:
            critic_input = 4* 5 # 4 features definidos en el entorno  por 5 carros (el agente y 4 más)
        self.m_critic = torch.nn.Sequential(
            torch.nn.Linear(critic_input, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, self.m_out),
            torch.nn.Softmax(dim=1)
        )
        self.m_critic.to(self.device)
        
        # para estabilidad del critic
        self.m_critic_target = torch.nn.Sequential(
            torch.nn.Linear(critic_input, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, self.m_out),
            torch.nn.Softmax(dim=1)
        )

        self.m_critic_target.load_state_dict(self.m_critic.state_dict())
        self.m_critic_target.to(self.device)  # Mover al dispositivo correcto

        self.critic_optimizador = torch.optim.Adam(self.m_critic.parameters(), lr=5e-4)
        #momento de gamma
        self.m_critic_scheduler =torch.optim.lr_scheduler.StepLR(self.critic_optimizador, step_size=1000, gamma=0.95)
        
        #debug info
        self.logger ={
            "loss":[], 
            "critic_loss":[],
            "model_loss":[],
            "reward_epocas":[],
            "estados":{}
        }
        self.distribucion_valores= None
        self.distribucion_step = 0
        self.distribucion_epsilon = 0
        self.distribucion_temp = 1.0
        self.distribucion_finaltemp=0.1 
    
    def debug_estados_info(self,key:str, valor:torch.Tensor)->None:
        """
        Guarda información de los estados en el logger del agente.
        Args:
            key (str): Clave para identificar el tipo de información.
            valor (torch.Tensor): Valor a guardar.
        """
        if key not in self.logger["estados"]:
            self.logger["estados"][key] = []
        if not isinstance(valor, torch.Tensor):
            self.logger["estados"][key].append(float(valor))
        else:
            self.logger["estados"][key].append(valor.detach().cpu().numpy().tolist())
    
    def debug_info(self, key:str, valor:float)->None:
        """
        Guarda información en el logger del agente.
        Args:
            key (str): Clave para identificar el tipo de información.
            valor (float): Valor a guardar.
        """
        if key not in self.logger:
            raise ValueError(f"llave '{key}' no válida. Llaves disponibles: {list(self.logger.keys())}")
        self.logger[key].append(valor)
    
    def save_debug_info(self,filename:str)->None:
        """
        Guarda la información del logger en un archivo json.
        Args:
            filename (str): Nombre del archivo donde se guardará la información.
        """
        with open(filename, 'w') as f:
            json.dump(self.logger, f, indent=4)
        print(f"Logger guardado en {filename}")
    
    def actuador(self,acciones:torch.Tensor)->torch.Tensor:
        """
        si todas las accioes tienen el mismo score, entonces elegir una al azar,
        de lo contrario, elegir la acción con el score más alto.
        """
        check = True
        size = acciones.shape[1] #suponniendo que es un tensor 1D (1,acciones)
        rounded_v = torch.round(acciones) if self.device !="mps" else acciones
        f= rounded_v[0,0]
        for i in range(1,size):
            if rounded_v[0,i] != f:
                check = False
                break
        if check:
            # si todas las acciones tienen el mismo score, elegir una al azar
            #print("\033[1;32mAcciones con el mismo score, eligiendo una al azar\033[0m")
            return torch.randint(0, 5, (1,)).item()
        else:
            # elegir la acción con el score más alto
            return acciones.argmax().item()

    def __getInfoFis__(self,fis_archivo:str)-> tuple[int, int, int]:
        FIS = fz.readfis(fis_archivo)
        inputs = len(FIS.Inputs)
        outputs = len(FIS.Outputs)
        rules = len(FIS.Rules)
        return FIS,inputs,outputs,rules
    
    def estado_velocidad(self,accion:int)->None:
        if accion == 3 and self.a_velocidad < 1: #FASTER
            self.a_velocidad += 1
        elif accion == 4 and self.a_velocidad > -1: #SLOWER
            self.a_velocidad -= 1
    
    def en_que_carril_voy(self,accion:int)->None:
        """
        Acciones:
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
        carriles:
            1: arriba
            -1: abajo
            0: medio
        """
        # si decide cambiar en carril en los extremos ( no se puede)
        if (self.a_carril == 1 and accion == 0) or (self.a_carril == -1 and accion == 2):
            return
        if(accion == 0):
            self.a_carril +=1
        elif(accion == 2):
            self.a_carril -=1
            
    def politica_exploracion(self,epsilon:float=0.1)->bool:
        """
        Determina si se explora o no explora
        Args:
            epsilon (float): probabilidad de explorar
        Returns:
            bool: True si se explora, False no explora
        """
        return torch.rand(1).item() < epsilon
    
    
    
    def acciones_raw(self,estado_actual:torch.Tensor,norm:bool=False)-> torch.Tensor:
        """Devuelve  la probabilidad de cada acción dada el estado actual del agente.
        Se tiene que hacer un preprocesamiento para que las entradas sean las del modelo.
        DCA = carro más cercano del carril de arriba
        DCM = carro más cercano del carril del medio
        DCA = carro más cercano del carril de abajo
        VE =  estado de la velocidad del agente
        V2:
        la entrada es el estado del entorno 
        carril, carrox1, carroy1, carrox2, carroy2, carrox3, carroy3, carrox4, carroy4, velocidad
        Args:
            estado_actual (torch.Tensor): estados del entorno actuales
        Returns:
            torch.Tensor: probabilidades de cada acción
        """
        if not self.config["raw_input"]:
            DCA,DCB,DCM = getLane(estado_actual,normalizado=norm)
            VE = estado_actual[0,2]
            #print(f"Input: carril:{self.a_carril}, DCA:{DCA:.3f}, DCB:{DCB:.3f}, DCM:{DCM:.3f}, VE:{VE:.3f}")
            estados = torch.tensor([[self.a_carril,DCA, DCB, DCM,VE]], dtype=torch.float32, device=self.device)
        else:
            estados = torch.tensor([self.a_carril]     +\
                estado_actual[1:,:-2].reshape(2*4).tolist() +\
                [self.a_velocidad], dtype=torch.float32, device=self.device)
        
        # Asegurar que estado_actual esté en el dispositivo correcto
        estado_actual = estado_actual.to(self.device)
        
        if self.config["world_data"]:
            #vectorial
            q_valores = self.m_critic(estado_actual.reshape(1,-1))
        else:
            #normal
            q_valores = self.m_critic(estados.unsqueeze(0))    
        
        acciones =self.model(estados)
        acciones = acciones.to(self.device)
        
        #exploración
        if self.politica_exploracion(self.config["explorar_porcentaje"]):
        #    # exploración aleatoria
        #    #print("\033[1;34mExplorando acciones aleatorias\033[0m")
        #    #evitar randomness
        #    temperatura =0.1
        #    num_acc = acciones.shape[1]
        #    #valores = torch.nn.Softmax(dim=0)(acciones)
        #    probabilidades = torch.rand_like(acciones)
        #    return (probabilidades/temperatura).reshape(1,num_acc)
            self.distribucion_step+=1
        self.set_distribucion(acciones)
        best_id = self.get_distribucion_sample()
        score = torch.zeros(1,self.m_out,device=self.device)
        score[0,best_id] = 1
        return score

    def get_distribucion(self,): 
        mejor = np.argmax(self.distribucion_valores.detach().cpu().numpy())
        #greedy
        #return torch.tensor(
        #    [1 if mejor == i else 0 for i in range(self.m_out)]
        #    ).to(torch.float32)
        #gready epsilon
        distribucion = [self.distribucion_epsilon/self.m_out for i in range(self.m_out)]
        distribucion[mejor] += 1 - self.distribucion_epsilon
        return torch.tensor(distribucion).to(torch.float32).to(self.device)
    
    def set_distribucion(self,valores):
        self.distribucion_valores = valores
        optimo = torch.argmax(valores, dim=1)
        self.distribucion_epsilon = self.distribucion_finaltemp + \
            (self.distribucion_temp-self.distribucion_finaltemp)* \
                np.exp(-self.distribucion_step/5000) 
        
    def get_distribucion_sample(self):
        distribucion = self.get_distribucion()
        acciones_lista = [i for i in range(self.m_out)]
        np_dist = distribucion.detach().cpu().numpy()
        return np.random.choice(acciones_lista,1, p=np_dist)[0]
    
    def step_train(self,x,target,critic_loss,loss_func=None):
        self.critic_optimizador.zero_grad()
        critic_loss.backward(retain_graph=True)
        #for param in self.m_critic.parameters():
        #    # truncar los gradientes para evitar fuga del gradiente
        #    if param.grad is not None:
        #        param.grad.data.clamp_(-1, 1)
        self.critic_optimizador.step()

        #for m_param in self.model.parameters():
        #    # truncar los gradientes para evitar fuga del gradiente
        #    if m_param.grad is not None:
        #        m_param.grad.data.clamp_(-1, 1) 
        
        m_loss = self.model.optimizador.step(x,target,loss_func)
        #m_loss = train_nfs(self.model,x,target,reward,1,1e-12,False)[0] #self.model.optimizador.step(x,target)
        
        #self.config["step"] = self.config["step"]+1 #% self.config["batch_size"])
        
        #if self.config["step"] % 100 == 0:
        self.m_critic_target.load_state_dict(self.m_critic.state_dict())

        self.debug_info("loss",m_loss)
        #self.debug_info("model_loss",m_loss)
        self.debug_info("critic_loss",critic_loss.item())   
        
    
    def retroalimentacion(self,
                        estado_actual:torch.Tensor,
                        estado_siguiente:torch.Tensor,
                        accion:int,
                        recompensa:float,
                        fin:int,
                        info:dict,
                        norm:bool=False,
                        batch_cont:bool = False) -> None:
        """
        Realiza una iteración del RLANFIS
        """
        
        #guardar en la memoria del agente el estado del entorno y las acciones del agente
        self.memoria.push(estado_actual,\
                            accion,\
                            self.a_carril,\
                            self.a_velocidad,\
                            recompensa,\
                            estado_siguiente,\
                            fin)
        #patch tipo de función de perdida que se usará en en el entrenamiento
        rl_loss_func = self.config["loss_func"]
        
        #batch es una lista de transiciones (estado,accion,carril_agente,recompensa,estado_siguiente,fin)
        batch = self.memoria.muestra_batch(self.config["batch_size"],batch_cont)
        
        #si ya se junta todo el batch se entrena
        if batch and self.config["entrenar"]:
            # preprocesar el batch para dejar carril_voy, DCA, DCB DCM,VE para el ANFISRL            
            pre_batch = []
            pre_batch_s = []
        
            reward_stack = []
            fin_stack = []

            target_stack = []
            accion_stack = []
            
            w_ea = torch.empty((0,20), dtype=torch.float32, device=self.device)
            w_es = torch.empty((0,20), dtype=torch.float32, device=self.device)
            raw_estado_batch = torch.empty((0,10), dtype=torch.float32, device=self.device)
            raw_estado_siguiente_batch = torch.empty((0,10), dtype=torch.float32, device=self.device)
            for transicion in batch:
                estado,t_accion,carril,ve,recompensa,estado_siguiente,fin = transicion
                if self.config["world_data"]:
                    w_ea = torch.concat((w_ea,estado.reshape(1,-1).to(self.device)))
                    w_es = torch.concat((w_es,estado_siguiente.reshape(1,-1).to(self.device)))

                tensor_input_muestra = torch.tensor([carril]+ estado[1:,:-2].reshape(2*4).tolist() + [ve], dtype=torch.float32, device=self.device)
                raw_estado_batch = torch.vstack((raw_estado_batch,tensor_input_muestra))
                
                
                carril_sig, _ = QueCarrilVoy(estado_siguiente[0,:].reshape(1,-1),norm=norm)
                if(carril_sig == None):
                    print(f"Estado: {estado_siguiente[0,:].reshape(1,-1)}",file=sys.stderr)
                tensor_input_muestra_siguiente = torch.tensor([carril_sig]+ estado_siguiente[1:,:-2].reshape(2*4).tolist() + [ve], dtype=torch.float32, device=self.device)
                raw_estado_siguiente_batch = torch.vstack((raw_estado_siguiente_batch,tensor_input_muestra_siguiente))
                
                reward_stack.append(recompensa)
                fin_stack.append(fin)
                tap = np.zeros(self.m_out)
                tap[t_accion] = 1.0 *recompensa
                target_stack.append(tap)
                
                accion_stack.append(t_accion)
                
                # preprocesar el estado actual
                ea_DCA,ea_DCB,ea_DCM = getLane(estado,normalizado=norm)
                ea_VE = ve #estado[0,2]
                ea = torch.tensor([[carril,ea_DCA, ea_DCB, ea_DCM,ea_VE]], dtype=torch.float32)
                pre_batch.append(ea.detach().cpu().numpy())
            
                # preprocesar el estado siguiente
                es_DCA,es_DCB,es_DCM = getLane(estado_siguiente,normalizado=norm)
                es_VE = ve
                es = torch.tensor([[carril,es_DCA, es_DCB, es_DCM,es_VE]], dtype=torch.float32)
                pre_batch_s.append(es.detach().cpu().numpy())
                
            # la red de valoracion recibe el batch de estados 
            tea = raw_estado_batch.clone() if self.config["raw_input"] else torch.tensor(np.array(pre_batch), dtype=torch.float32, device=self.device).squeeze()
            tes = raw_estado_siguiente_batch.clone() if self.config["raw_input"] else torch.tensor(np.array(pre_batch_s), dtype=torch.float32, device=self.device).squeeze()
            teay = torch.tensor(np.array(target_stack), dtype=torch.long, device=self.device).squeeze()
            ta = torch.tensor(np.array(accion_stack), dtype=torch.long, device=self.device).squeeze()
            
            
            estado_valor_accion = self.model(tea)
            #ajustar dimensiones
            tdim = teay.clone() if teay.shape == estado_valor_accion.shape else teay.unsqueeze(0)
            estado_valor_accion = estado_valor_accion.gather(1, tdim).squeeze().to(torch.float32)
            
            if self.config["world_data"]:
                w_ea = w_ea.reshape(-1,20)
                w_es = w_es.reshape(-1,20)
                #estado_valor_accion = self.m_critic(w_ea) #self.model(tea).to(torch.float32) #self.m_critic(w_ea) #
                valor_siguientes_accion = self.m_critic(w_es)#self.m_critic_target(w_es)
            else:
                #estado_valor_accion = self.m_critic(tea) #self.model(tea).to(torch.float32) #self.m_critic(tea)
                # la red de valoracion recibe el batch de estados siguientes
                valor_siguientes_accion = self.m_critic(tes) #self.m_critic_target(tes).squeeze()
            
            estado_valor_siguientes_accion = torch.zeros_like(valor_siguientes_accion,dtype=torch.float32, device=self.device)
            
            
            # calcular el target de la valoración de las acciones
            tr = torch.tensor(reward_stack, dtype=torch.float32, device=self.device).unsqueeze(1)
            tf = torch.tensor(fin_stack, dtype=torch.long, device=self.device).unsqueeze(1)
            #sin_fin = tf[teay!=1]
            sin_fin = (tf == 0).to(self.device)
            sin_idle = (ta!=1).to(self.device)
            
            sin_finidle = torch.tensor([True if (sin_fin[i] and sin_idle[i]) else False for i in range(len(sin_fin))], device=self.device)
            
            if sin_fin.sum() == 0:
                return # no hay estados validos, no se entrena 
            
            ids_sinfin = torch.where(sin_fin)[0]
            ids_sinidle = torch.where(sin_finidle)[0]
            
            #sin idle?
            #ids_sinfin = ids_sinfin[teay[ids_sinfin] != 1]
            
            argmax_vsa = valor_siguientes_accion.argmax(dim=1)

            for id in ids_sinfin:
                estado_valor_siguientes_accion[id,argmax_vsa[id]] = 1.0

            #estado_valor_siguientes_accion[ids_vsa] = valor_siguientes_accion[ids_sinidle]
            
            

            #q_valores_actuales = estado_valor_accion.squeeze()#.gather(1, teay).squeeze() 
            
            #with torch.no_grad():
                # calcular el target de la valoración de las acciones
                # target = r + gamma * max(Q(s',a'))
                #estado_valor_siguientes_accion = self.m_critic_target(tes if not self.config["world_data"] else w_es)
                #max_q_siguientes = estado_valor_siguientes_accion.max(dim=1)[0]
            target_q_valor = tr+self.config["gamma"]*estado_valor_siguientes_accion#.reshape(-1,1) 
            
            #normalizar el target que queden los scores entre 0 y 1
            target_q_valor = torch.nn.functional.sigmoid(target_q_valor) 
            
            #actualizar critic
            critic_loss = rl_loss_func(estado_valor_accion[ids_sinidle], target_q_valor[ids_sinidle])
            
        
            
            #actualizar política
            #act_probs= self.model(tea)
            #log_probs = torch.log_softmax(act_probs, dim=1)
            #select_log_probs = log_probs[range(batch_cont),act_probs]
            
            #reward_discounted = tr*self.config["gamma"]*(1-tf)
            #target_estado_valor_accion =reward_discounted+estado_valor_siguientes_accion.squeeze()
            # calcular el loss con respecto al target de las accciones y los estados
            #critic_loss = mse_loss(estado_valor_accion.squeeze(),target_estado_valor_accion.squeeze())

            #pred_y = self.model(tea).to(torch.float32)
            #TD error
            #td_error= target_q_valor[ids_sinfin,]- pred_y[ids_sinfin] #q_valores_actuales[ids_sinfin,] #[range(len(ta)),ta]

            #for i in range(len(teay)):
            #    lr = torch.tanh(td_error[i]*1e-3)
            #    # actualizar el valor de la acción en el estado actual
            #    pred_y[i,teay[i]] += lr 
            #    #normalizar 
            #    pred_y[i] = torch.softmax(pred_y[i], dim=0)
            
            
            
            #T = target_estado_valor_accion
            tsf = target_q_valor[ids_sinfin].clone().to(torch.float32) #teay[ids_sinfin].clone() #target_q_valor.clone().to(torch.float32)#[ids_sinfin] #target_sugerencia(tea[ids_sinfin,:-1],5) #target_q_valor[ids_sinfin].clone() #ta[ids_sinfin].clone()
            isf = tea[ids_sinfin].clone().to(torch.float32)   #[ids_sinfin].clone()
            
            #model_loss = critic_loss + ((target_q_valor[ids_sinfin]-estado_valor_accion[ids_sinfin]).sum())
            
            self.step_train(isf,tsf,critic_loss,rl_loss_func)#td_error)
            

    
    def mostrarHeatMapAcciones(self,prob_acciones:torch.Tensor,id_accion:int)->None:
        plt.clf()
        # Mover tensor a CPU para visualización
        muestra = prob_acciones.detach().cpu().numpy()
        # Desactiva la barra de herramientas
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams["figure.raise_window"]=False
        # Configura el tamaño de la figura
        plt.rcParams['figure.figsize'] = (6, 2)  
        ax = sns.heatmap(muestra.reshape(1,-1),annot=True,cmap="mako",cbar=False)
        fig = ax.get_figure()
        #remarcar el valor más alto
        max_id = id_accion
        ax.add_patch(Rectangle((max_id,0),1,1, fill=True, facecolor='orange', edgecolor='orange', lw=3))
        #labels
        plt.xticks(ticks=[0.5,1.5,2.5,3.5,4.5], labels=["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"])
        #plt.xlabel("Acciones")
        plt.ylabel("Score Acciones")
        plt.show(block=False)
        plt.pause(0.001)  # Pausa para mostrar la figura


def target_sugerencia(estados:torch.Tensor,num_acciones:int=5,raw=False, device=None)-> torch.Tensor:
    """
    da una sugerancia de la posible acción a tomar dependiendo de un estado dado.
    Args:
        estados (torch.Tensor): tensor de estados del entorno preprocesados en carril_agante,DCA, DCB, DCM\n
            de dimensiones (batch,4)
        num_acciones (int): número de acciones posibles
        device: dispositivo donde crear el tensor
    Returns:
        torch.Tensor: tensor de score por acciones de tamaño (batch,num_acciones)
    """
    # Determinar el dispositivo
    if device is None:
        device = estados.device if hasattr(estados, 'device') else torch.device('cpu')
    
    carril = estados[:, 0].clone().to(torch.long)
    scores = torch.zeros((estados.shape[0],num_acciones), dtype=torch.float32, device=device)
    maximos = torch.argmax(estados[:,1:], dim=1)
    minimos = torch.argmin(estados[:,1:], dim=1)
    for i in range(estados.shape[0]):
        if maximos[i] == minimos[i]: #libre
            scores[i,num_acciones-1]= 1.0
        elif maximos[i] == 0 and carril[i] == 0: #carril arriba más libre
            scores[i,0] = 1.0
        elif (maximos[i] == 0 and carril[i] == 1) or (maximos[i]==1 and carril[i] == -1):
            scores[i,1] = 1.0
        elif maximos[i] == 1 and carril[i] == 0: #carril abajo más libre
            scores[i,2] = 1.0
        elif maximos[i] == 2 and carril[i] == 1: #carril medio más libre y voy arriba
            scores[i,2] = 1.0
        elif maximos[i] == 2 and carril[i] == -1: #carril medio más libre y voy abajo
            scores[i,0] = 1.0
    return scores
def _inicio_timeit():
    return time.time()

def _fin_timeit(inicio:float)->float:
    return time.time()-inicio

def eliminar_archivos_part():
    """
    Elimina todos los archivos en el directorio actual cuyo nombre contenga 'part-'.
    """
    cwd = os.getcwd()
    for archivo in os.listdir(cwd):
        if "part-" in archivo and os.path.isfile(os.path.join(cwd, archivo)):
            try:
                os.remove(os.path.join(cwd, archivo))
                print(f"Archivo eliminado: {archivo}")
            except Exception as e:
                print(f"No se pudo eliminar {archivo}: {e}")

if __name__ == "__main__":
    import gymnasium as gym
    import highway_env
    import pprint
    from utils.funciones_auxiliares import QueCarrilVoy
    import datetime
    #funciones de perdidas
    from torch.nn.functional import mse_loss, cross_entropy, l1_loss, huber_loss, smooth_l1_loss


    device = "cpu" #"cuda" if torch.cuda.is_available() else  "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\033[1;36m=== Configuración del Sistema ===\033[0m")
    print(f"Dispositivo principal: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Memoria GPU libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
    print(f"\033[1;36m================================\033[0m\n")

    # Ejemplo de uso
    normalizar = True
    env = gym.make('highway-v0', 
            render_mode="rgb_array", #rgb_array #human
            config={
                "observation": {
                    "type": "Kinematics", # "OccupancyGrid"
                    'simulation_frequency': 60,
                    #'vehicles_count': 10,
                    'vehicles_density': 2,
                    "manual_control": False,
                    "show_trajectories":False,
                    'offroad_terminal': True,
                    'offscreen_rendering': True,
                    #"duration":3
                    #"controlled_vehicles": 1,
                    #"vehicles_count": num_carros,  # matriz de entrada de los carros
                    "features": [
                        "x", 
                        "y", 
                        "vx", 
                        "vy",
                        #"heading", 
                        #"long_off", 
                        #"lat_off",
                        #"ang_off"
                        ],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                        #"heading": [-1, 1],
                        #"long_off": [-100, 100],
                        #"lat_off": [-100, 100],
                    },
                    "absolute": False,
                    "order": "sorted",
                    "normalize": normalizar,
                },
                "lanes_count": 3,
            }
        )
    #env.unwrapped.config["vehicles_count"]=50
    env.unwrapped.config["vehicles_density"]=1.3 #1.5 es mucho
    env.reset()
    pprint.pprint(env.unwrapped.config)
    bs =128 #32 #64 #128 #256 #512 #1024
    ms = 1024**3 #*10 #500mb #1024 #1GB 
    
    raw_input = False 
    fis_cargar = "HWR_rl_workstation_workstation_1000000.fis" #"HWR_rl.fis" if raw_input else "HWB_rl.fis"#"HWB_rl.fis"
    fis_guardar = fis_cargar.split(".")[0]
    print(f"\033[1;36mCargando FIS: {fis_cargar}\033[0m")
    #device = torch.device("cuda") #cuda mps cpu
    
    agente = Agente(fis_cargar,batch_s=bs,w_pre=True,memoria_size=ms,mu_dec=0.1,mu_inc=10,mu_max=1e10,device=device)
    agente.config["entrenar"] = False
    agente.config["explorar_porcentaje"] = 0.20 if agente.config["entrenar"] else 0.0
    agente.config["raw_input"] = raw_input # si se procesa la entrada del entorno o no
    agente.config["loss_func"] = l1_loss #loss_cosine_similarity #l1_loss # mse_loss # cross_entropy # huber_loss # smooth_l1_loss
    epocas =10_000 if agente.config["entrenar"] else 100
    lotes_continuos = False
    renderizar = not agente.config["entrenar"]
    
    tiempo_inicio_ejecucion = _inicio_timeit()
    
    i=0
    for episodio in range(epocas):
        tiempo_inicio_episodio = _inicio_timeit()
        if(episodio+1) % int(epocas*.1) == 0:
            agente.config["explorar_porcentaje"] -=0.015
            
            if agente.config["entrenar"]:
                #guardar temporalmente por si se va la luz
                version  = datetime.datetime.now().strftime("%Y%m%d") 
                GuardarFIS(f"{fis_guardar}{version}_part-{i+1}.fis",fis=agente.m_fis,modelo=agente.model)
                agente.save_debug_info(f"agente_debug_info_{version}_part-{i}.json")
                i+=1
            
        obs, info = env.reset()
        done = False
        r_e=0
        agente.a_carril,_ = QueCarrilVoy(obs[0:1,:],norm=normalizar)
        while not done:
            
            #
            if(episodio+1) % int(epocas*.1) == 0 and not renderizar:
                env.render()
                
            elif renderizar: #nunca se cierra la ventana
                env.render()
                #print("carril actual:",agente.a_carril)
                print(f"observaciones:\n", obs[0:5,:])
            else:
                env.close()
            
            acciones = agente.acciones_raw(torch.tensor(data=obs, dtype=torch.float32,device=device),normalizar)
            
            accion = agente.actuador(acciones) #acciones.argmax().item()
            
            if(episodio+1) % int(epocas*.1) == 0 and not renderizar:
                agente.mostrarHeatMapAcciones(acciones,accion)
            elif renderizar: #nunca se cierra la ventana
                agente.mostrarHeatMapAcciones(acciones,accion)
            else:
                plt.close()
            
            obs_sig,reward,done,truncated,info = env.step(accion)
            agente.en_que_carril_voy(accion)
            agente.estado_velocidad(accion)
            agente.retroalimentacion(torch.tensor(obs, dtype=torch.float32,device=device),
                                    torch.tensor(obs_sig, dtype=torch.float32,device=device),
                                    accion,
                                    reward,
                                    done,
                                    info,
                                    normalizar,
                                    lotes_continuos)
        
            obs = obs_sig
            r_e += reward
            #agente.debug_info("acciones",accion)
            #agente.debug_info("carril",agente.a_carril)
            agente.debug_estados_info("reward_episodio_"+str(episodio+1),reward)
            agente.debug_estados_info("episodio_"+str(episodio+1),torch.tensor(obs, dtype=torch.float32))
            agente.debug_estados_info("acciones_episodio_"+str(episodio+1),accion)
            agente.debug_estados_info("ve_episodio_"+str(episodio+1),agente.a_velocidad)

        #fin while
        agente.debug_info("reward_epocas",r_e)
        
        
        if (episodio+1) % int(epocas*.25) == 0 and agente.config["entrenar"]:
            print("\033[1;37mCheckPoint\033[0m")
            GuardarFIS(f"{fis_guardar}.fis",fis=agente.m_fis,modelo=agente.model)
            
        fin_tiempo_episodio = _fin_timeit(tiempo_inicio_episodio)
        e_minutos = int(fin_tiempo_episodio // 60)
        e_segundos = fin_tiempo_episodio % 60
        stats =f"REP:{r_e:.2f} explorar: {agente.config['explorar_porcentaje']:.2f} | " + f"Tiempo: {e_minutos} min {e_segundos:.2f} segs"
        print(f"\033[1;33m[Episodio {episodio+1}/{epocas}]\033[0m \033[1;34mmem[{len(agente.memoria)}/{agente.memoria.capacidad}] \033[1;35m{stats}\033[0m")
    
    if agente.config["entrenar"]:
        version  = datetime.datetime.now().strftime("%Y%m%d%H%M%S") 
        GuardarFIS(f"{fis_guardar}.fis",fis=agente.m_fis,modelo=agente.model)
        agente.save_debug_info(f"agente_debug_info_{version}.json")
        #agente.memoria.guardar_memoria(f"agente_memoria_{version}.pkl")
        eliminar_archivos_part()
    
    fin_tiempo_ejecucion = _fin_timeit(tiempo_inicio_ejecucion)
    horas = int(fin_tiempo_ejecucion // 3600)
    hres = fin_tiempo_ejecucion % 3600
    minutos = int(hres // 60)
    mres  = hres % 60
    segundos = int(mres // 60)
    print(f"\033[1;32mTiempo total de ejecución: {horas}:{minutos}:{segundos}\033[0m")
    env.close()
            
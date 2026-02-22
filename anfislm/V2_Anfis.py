import torch
import torch.nn as nn
from torch.autograd import grad
import itertools
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from utils.Anfis_utils import CargarFIS
from utils.funciones_auxiliares import PrintLogLevel

#clase abstracta para el optimizadores
class Optimizador:
    def __init__(self,device:str='cpu'):
        self.device = device
    def step(self):
        raise NotImplementedError("El optimizador no tiene implementado el método")
    def setParams(self,*args):
        raise NotImplementedError("El optimizador no tiene implementado el método")

class CapaGaussiana(nn.Module):
    def __init__(self, n_in, k_reglas, ant:np.array=None,device:str='cpu'):
        """
        @param n_in: número de entradas (N)
        @param k_reglas: número de membresías por cada entrada y reglas (K)
        @param self.centro, self.sigma tienen forma (n_in, k_reglas)
        @param ant = valores de las membresias extraidos del FIS
        """
        super(CapaGaussiana, self).__init__()
        self.n_in = n_in
        self.reglas = k_reglas
        # MPS does not support float64, use float32 for MPS
        dtype = torch.float32 if device == 'mps' else torch.float64

        self.centro = nn.Parameter(torch.zeros(n_in, k_reglas, dtype=dtype, device=device))
        #evitar division entre cero
        self.sigma = nn.Parameter(torch.zeros(n_in, k_reglas, dtype=dtype, device=device) + 0.1)

        if(ant is not None):
            #cargar los valores de las membresias del ant
            #(num_in, num_reglas, 2) por gausiana el 2
            self.__ExtraerMembresias(ant)

            print(f"sigma: {self.sigma.shape}")
            print(f"centro: {self.centro.shape}")

    def __ExtraerMembresias(self,datos:np.array)->None:
        num_in = self.n_in
        num_reglas = self.reglas
        #print(f"[ExtraerMembresia GaussianLayer] {num_in} ,{num_reglas}")
        for i in range(num_in):
            for r in range(num_reglas):
                self.centro.data[i,r]= torch.tensor(datos[i,r,1], dtype=self.centro.dtype)
                self.sigma.data[i,r]= torch.tensor(datos[i,r,0], dtype=self.sigma.dtype)

    def forward(self, x):
        """
        @param x: (muestras, n_in)
        Salida: (muestras, k_reglas)
          membership_vals[:, i, j] = exp(-0.5*((x_i - mu[i, j]) / sigma[i, j])^2)
        """
        # x.shape => (muestras, n_in)
        # centro.shape => (n_in, k_reglas) => expandimos a (1, n_in, k_reglas)

        x_exp = x.unsqueeze(-1)                     # (muetras, n_in, 1)
        mu_exp = self.centro.unsqueeze(0)           # (1, n_in, k_reglas)
        sigma_exp = self.sigma.unsqueeze(0)         # (1, n_in, k_reglas)

        # Gaussiana
        # (muestras, n_in, k_reglas)
        diff = (x_exp-mu_exp)/sigma_exp
        membership_vals = torch.sum(diff**2,dim=1)  # [muestras, k_reglas]
        #print(f"memb: {membership_vals.shape}")
        return membership_vals                      # (muestras, k_reglas)

class CapaFuerzaDisparo(nn.Module):
    def __init__(self, n_in, k_reglas,muestras=719):
        """
        @param n_in: N
        @param k_reglas: K
        la combinacion de las reglas es lineal ya que por
        cada entrada es una reglla ejemplo.
        1 1 1
        2 2 2
        3 3 3
        por lo que podemos hacer el producto como la fuerza de disparo y por la
        propiedad de los exponenciales seria la sumatoria (recibe por parametro)
        """
        super(CapaFuerzaDisparo, self).__init__()
        self.n_in = n_in
        self.k_reglas = k_reglas #reglas


    def forward(self, membership_vals):
        """
        membership_vals: (muestras,k_reglas)
        Retorna un tensor (muestras, k_rules) 
        donde por cada entrada existe una regla
        1 1 1 1
        2 2 2 2
        3 3 3 3 
        """

        ALPHA = torch.exp(-0.5*membership_vals)     # (muestras, k_reglas)
        #print(f"fd: {ALPHA.shape}")
        rule_activations = ALPHA
        return rule_activations                     # (muestras, k_reglas)


class NormalLayer(nn.Module):
    def __init__(self):
        """
        Normaliza la fuerza de dispparo
        alpha/sum(alpha)
        """
        super(NormalLayer, self).__init__()

    def forward(self, rule_activations):
        phi = rule_activations/torch.sum(rule_activations,dim=1,keepdim=True)

        return phi                          # (muestras, reglas)


class CapaCenterOfSets(nn.Module):
    def __init__(self, num_rules, n_out, con:np.array=None,device='cpu'):
        """
        @param num_rules 
        @param n_out 
        @param con = valores de los consecuentes extraidos del FIS
        """
        super(CapaCenterOfSets, self).__init__()
        self.num_rules = num_rules
        self.n_out = n_out
        # MPS does not support float64, use float32 for MPS
        dtype = torch.float32 if device == 'mps' else torch.float64

        # (num_rules, n_out)
        self.centers = nn.Parameter(torch.zeros(num_rules, n_out, dtype=dtype, device=device))
        if(con is not None):
            #cargar las membresias del fis (num_out, num_reglas, 1) ya que es constante
            self.__ExtraerMembresias(con)
            print(f"theta: {self.centers.shape}")

    def __ExtraerMembresias(self,data:np.array) -> None:
        num_out = self.n_out
        num_reglas = self.num_rules
        for i in range(num_out):
            for r in range(num_reglas):
                self.centers.data[r,i] = torch.tensor(data[i,r,0], dtype=self.centers.dtype)

    def forward(self, rule_activations):
        """
        rule_activations: (muestras, num_rules)
        Retorna: (muestras, n_out)
        y = fuerza_normalizada . constantes(centros)
        """

        centers_exp = self.centers.unsqueeze(0)             # (1, num_rules, n_out)
        #print(self.centers.shape,"@",rule_activations.shape)
        output = rule_activations @ self.centers
        return output


class RLANFISBuilder:
    def __init__(self):
        self.anfis = None
        self.fis = None
        self.ins = None
        self.out = None
        self.reglas = None
        self.mu_inc = 10
        self.mu_dec = 0.1
        self.mu_max = 1e10
        self.valmaxfails = 20
        self.tipo = "regresion"
        self.optimizador = None
        self.device = None

    def AddDevice(self, device:str):
        if device not in ['auto', 'cpu', 'cuda','mps']:
            raise ValueError("device debe ser 'auto', 'cpu', 'cuda' o 'mps'")
        self.device = device
        return self
    
    def AddFIS(self, fis:str):
        self.fis = fis
        return self

    def AddInputs(self, i:int):
        self.ins = i
        return self
    
    def AddOutputs(self, o:int):
        self.out = o
        return self
    
    def AddRules(self, r:int):
        self.reglas = r
        return self

    def AddMuStats(self, mu_dec:float, mu_inc:float, mu_max:float):
        self.mu_dec = mu_dec
        self.mu_inc = mu_inc
        self.mu_max = mu_max
        return self
    
    def AddValMaxFails(self, valmaxfails:int):
        self.valmaxfails = valmaxfails
        return self 
    
    def AddTipoProblema(self,tipo:str):
        self.tipo = tipo
        return self
    
    def AddOptimizador(self, optimizador:type):
        if not (isinstance(optimizador, type) and issubclass(optimizador, Optimizador)):
            raise TypeError("optimizador_cls debe ser una subclase de Optimizador")
        self.optimizador = optimizador
        return self
    
    def Build(self):
        if(self.fis==None or self.ins==None or self.out==None or self.reglas==None):
            raise Exception("Para construir el modelo se ocupa como minimo el fis, numero de entradas, salidas y reglas")

        self.anfis = ANFISND(self.ins,self.out,self.reglas,self.fis,
                            self.mu_inc,self.mu_dec,self.mu_max,
                            self.valmaxfails,
                            self.tipo,
                            self.optimizador,
                            device=self.device)
        return self.anfis

class ANFISND(nn.Module):
    def __init__(self, n_in, n_out, k_reglas, fis_path:str=None, muinc=10,mudec=0.1,mumax=1e50,valmaxfails=20,
                 tipo:str="regresion",optimizador:Optimizador=None,device:str='auto'):
        """
        n_in: N
        n_out: M
        k_reglas: K
        """
        super(ANFISND, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.k_reglas = k_reglas
        #pal optimizador
        self.mu=0.01
        self.mu_inc = muinc
        self.mu_dec = mudec
        self.mu_max = mumax
        self.num_fallos=0
        self.num_max_fallos = valmaxfails
        self.fis =None
        self.tipo = tipo
        if device != 'auto':
            self.device = device 
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        a_mem=None
        c_mem=None
        if(fis_path is not None):
            # regresa una tupla (fis, (num_in,num_reglas,2), (num_out, num_reglas,1))
            fis,a_mem,c_mem = CargarFIS(fis_path)

        self.membership_layer = CapaGaussiana(n_in, k_reglas, a_mem,self.device)
        self.rule_layer = CapaFuerzaDisparo(n_in, k_reglas)
        self.normal_layer = NormalLayer()

        num_rules = k_reglas 
        self.defuzz_layer = CapaCenterOfSets(num_rules, n_out, c_mem,self.device)
        
        self.optimizador = optimizador(self,device=self.device)
        self.optimizador.setParams(0.01,self.mu_dec,self.mu_inc,self.mu_max)
        
        # Move model to device after constructing all layers
        self.to(self.device)
        

    def forward(self, x):
        """
        x: (muestras, n_in)
        """
        membership_vals = self.membership_layer(x)          # (muestras, k_reglas)
        #print(f"[{membership_vals.shape}]")
        rule_acts = self.rule_layer(membership_vals)        # (muestras, k_reglas)
        #print(f"[{rule_acts.shape}]")
        norm_acts = self.normal_layer(rule_acts)            # (muestras, k_reglas)
        #print(f"[{norm_acts.shape}]")
        y = self.defuzz_layer(norm_acts)                    # (batch_size, n_out)
        #print(f"[{y.shape}]")
        #return y
        # en clasificación no tener valores negativos en la salida 
        # ya que representan la probabilidad de que sea el objeto
        return y #torch.nn.functional.softmax(y,dim=1) if self.tipo=="clasificacion" else y


class LevenberMaquardtOpt(Optimizador):
    def __init__(self,model,lambda_init=0.01,lambda_decr=0.9,lambda_incr=10,device='cpu')->None:
        super().__init__(device=device)
        self.model = model
        self.lambda_val = lambda_init
        self.lambda_decr = lambda_decr
        self.lambda_incr = lambda_incr
        self.lambda_max = 1e10
        self.device = device
        print(f"LBM -> {device}")

        self.params = list(model.parameters())
        self.num_params = sum(p.numel() for p in self.params)
    
    def setParams(self, *args):
        #super().setParams(*args)
        #print(args)
        self.lambda_val = args[0]
        self.lambda_decr = args[1]
        self.lambda_incr = args[2]
        self.lambda_max = args[3]
    
    def _get_param_vector(self)->torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.params])
    
    def _set_param_vector(self, vector:torch.Tensor)->None:
        id = 0
        for param in self.params:
            param_size = param.numel()
            param.data = vector[id:id+param_size].view_as(param.data)
            id+= param_size
    
    def _compute_error(self, Y_pred:torch.Tensor, Y_true:torch.Tensor)->torch.Tensor:
        error_vec=[]
        for t,y in zip(Y_true,Y_pred):
            error = t-y
            error_vec.append(error.view(-1))
        # MPS does not support float64, use float32 for MPS
        dtype = torch.float32 if self.device == 'mps' else torch.float64
        return torch.cat(error_vec).to(dtype)
    
    
    
    def jacobiana(self,X,Y) -> tuple[torch.Tensor,torch.Tensor]:
        parametros_vec = self._get_param_vector()
        out = self.model(X)
        curr_error = self._compute_error(out,Y)
        
        error_size = curr_error.numel() #sum(y.numel() for y in Y)
        # MPS does not support float64, use float32 for MPS
        dtype = torch.float32 if self.device == 'mps' else torch.float64
        jacob = torch.zeros(error_size,self.num_params,dtype=dtype,device=torch.device(self.device))
        
        # Diferencias Finitas - use larger epsilon for float32 stability
        epsilon = 1e-4 if self.device == 'mps' else 1e-8
        for i in range(self.num_params):
            p_params = parametros_vec.clone()
            p_params[i] += epsilon
            self._set_param_vector(p_params)
            
            p_out = self.model(X)
            p_error = self._compute_error(p_out,Y)
            
            jacob[:,i] = (p_error - curr_error) / epsilon
        
        self._set_param_vector(parametros_vec)
        
        return jacob, curr_error
    
    def step(self,X,Y):
        jacobiana, error_vec = self.jacobiana(X,Y)
        
        #JTJ
        JtJ = torch.matmul(jacobiana.t(), jacobiana) 
        # MPS does not support float64, use float32 for MPS
        dtype = torch.float32 if self.device == 'mps' else torch.float64
        diag_JtJ = torch.eye(jacobiana.shape[1], dtype=dtype, device=torch.device(self.device))
        g = torch.matmul(jacobiana.t(), error_vec)
        
        #parametros
        c_params = self._get_param_vector()
        
        #JJ = JtJ.clone()
        #JJ[I,I] += self.lambda_val
        A = JtJ + self.lambda_val * diag_JtJ
        try:
            #(JtJ+lambdaDiag)^-1 = Je
            delta = -torch.linalg.solve(A, g)
        except Exception as e:
            #print(f"[LBM ERROR] solve failed: {e}")
            delta = -torch.zeros_like(g)

        # Check if delta contains meaningful updates
        delta_norm = torch.norm(delta)
        #if delta_norm < 1e-10:
        #    PrintLogLevel("WARNING",f"delta muy pequeña para float32: {delta_norm:.2e}")

        #actualizar params
        n_params = c_params + delta
        self._set_param_vector(vector=n_params)
        
        n_out = self.model(X)
        n_error = self._compute_error(n_out,Y)
        
        c_loss = torch.sum(error_vec**2)
        n_loss = torch.sum(n_error**2)
        
        if(n_loss < c_loss):
            self.lambda_val *= self.lambda_decr
            return n_loss.item()
        else:
            self.lambda_val *= self.lambda_incr
            self._set_param_vector(c_params)
            return c_loss.item()
    
    


def train_nfs(model, X_train, y_train, epochs=100,tolerancia=1e-6,debug=False):
    """
    Train the neuro-fuzzy system using Levenberg-Marquardt optimization
    """
    PrintLogLevel("INFO",f"Inputs: {X_train.shape} , Outputs: {y_train.shape}")
    optimizer = model.optimizador 

    losses = []
    
    for epoch in range(epochs):
        loss = optimizer.step(X_train, y_train)
        losses.append(loss)

        if torch.isnan(torch.tensor(loss)):
            PrintLogLevel("ERROR",f"Loss es NaN en epoch {epoch+1}.")
            return losses
        
        if (epoch % int(epochs*.1) if epochs >100 else 10) == 0 and debug:
            PrintLogLevel("INFO",f"Epoch {epoch}, Loss: {loss:.9f}")
        
        if(loss <= tolerancia):
            PrintLogLevel("WARNING",f"Se llego a la tolerancia {loss:.9f} <= {tolerancia}")
            return losses
        
        if(optimizer.lambda_val > optimizer.lambda_max):
            PrintLogLevel("ERROR",f"[{epoch+1}] El modelo llego a las mu maximas {optimizer.lambda_val:.1E} >= {optimizer.lambda_max:.1E}({optimizer.lambda_val>=optimizer.lambda_max})")
            PrintLogLevel("ERROR",f"[{epoch+1}] con un loss de {loss:.9f}")
            return losses
    
    return losses


def CapaCompetitiva(X)->torch.Tensor:
    indices = X.argmax(dim=1)
    return torch.nn.functional.one_hot(indices,num_classes=X.shape[1])


def gx_train(modelo:nn.Module, loss:torch.Tensor,YH:torch.Tensor,T:torch.Tensor,X_VAL:torch.Tensor,Y_VAL:torch.Tensor,learning_rate=1) -> None:
    grads = grad(loss, modelo.parameters(), create_graph=True)

    with torch.no_grad():
        for p in modelo.parameters():
            numel = p.numel()
            update = p.view(p.shape) #delta[idx: idx+numel].view(p.shape)
            #print(f"update shape: {update.shape}")
            p -= learning_rate * update
            idx += numel

def lm_train(modelo:nn.Module, loss:torch.Tensor,YH:torch.Tensor,T:torch.Tensor,X_VAL:torch.Tensor,Y_VAL:torch.Tensor,learning_rate=1,tipo="regresion") -> None:
    error_vec = (YH - T).view(-1)
    # MPS does not support float64, use float32 for MPS
    dtype = torch.float32 if str(error_vec.device) == 'mps' else torch.float64
    error_vec = error_vec.to(dtype)

    J_rows = []
    for i in range(len(error_vec)):
        e_i = error_vec[i]
        grad_i = grad(e_i, modelo.parameters(), retain_graph=True)
        row_i = []
        for gi in grad_i:
            row_i.append(gi.view(-1))
        row_i = torch.cat(row_i)
        #print(f"\033[33mrow_i: {row_i.shape}\033[0m")
        J_rows.append(row_i.unsqueeze(0))

    J = torch.cat(J_rows, dim=0)  
    # A = (J^T J + λ diag(J^T J)), g = J^T error
    JTJ = J.t() @ J
    diag_JTJ = torch.eye(J.shape[1], dtype=J.dtype, device=J.device)
    gxNorm = torch.norm(2*J.t()@error_vec)

    while(modelo.mu <= modelo.mu_max):
        A = JTJ + modelo.mu * diag_JTJ
        #print(f"{J.t().shape} @ {error_vec.shape}")
        g = J.t() @ error_vec
        try:
            delta = -torch.linalg.solve(A, g)
        except RuntimeError:
            delta = -torch.zeros_like(g)


        ytest = modelo(X_VAL)
        if(tipo=="clasificacion"):
            #ytest = CapaCompetitiva(ytest)
            n = T.shape[0]
            clamp_probs = torch.clamp(ytest,min=1e-8,max=1.0-1e-8)
            perf1 = -(1/n)*torch.sum(Y_VAL*torch.log(clamp_probs))
        else:
            perf1 =torch.sum((ytest - Y_VAL)**2)
        # Actualizamos parámetros
        idx = 0
        estado_temporal= copy.deepcopy(modelo)
        with torch.no_grad():
            for p in modelo.parameters():
                numel = p.numel()
                update = delta[idx: idx+numel].view(p.shape)
                #print(f"update shape: {update.shape}")
                p += learning_rate * update
                idx += numel
        
        #cambiar a validacion
        ytest = modelo(X_VAL)
        if(tipo=="clasificacion"):
            #ytest = CapaCompetitiva(ytest)
            clamp_probs = torch.clamp(ytest,min=1e-8,max=1.0-1e-8)
            perf2 = -(1/n)*torch.sum(Y_VAL*torch.log(clamp_probs))
        else:
            perf2 =torch.sum((ytest - Y_VAL)**2)
        #print(f"[{loss.item()} > {etes.item()}]")
        if(perf2.item()< perf1.item()):
            if(modelo.mu>1e-300): # por si llega al num maximo en python
                modelo.mu*=modelo.mu_dec
            return #break
        modelo.load_state_dict(estado_temporal.state_dict())
        modelo.mu*=modelo.mu_inc
        #print(f"[{modelo.mu:.1e}] incrementar MU")
        #fin de while
    #fin de lm_train

# -------------------------------------------------------------------
if __name__ == "__main__":
    pass


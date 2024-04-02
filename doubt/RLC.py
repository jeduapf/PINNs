import tensorflow as tf
import torch
from torchsummary import summary
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import random
import plotly.graph_objects as go
import mat73

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ANN(torch.nn.Module):
    '''
        Defining the architecture of the Neural Network.
        According to https://github.com/maziarraissi/PINNs.

        Ex.: F(t,a,b,c,d,e) = (g(t,a,b,c,d,e), h(t,a,b,c,d,e), v(t,a,b,c,d,e))

        So there would be 6 neurons in the input layer and 3 neurons as the output layer
        Since there are 6 entries (t,a,b,c,d,e) and 3 output functions (g,h,v)

        Thus layers = [6,20,20,20,20,20,20,20,20,3]

        Input:
            - inputs[int]: First layer neurons => Input Space of the function
            - outputs[int]: Last layer neurons => Output Space of the function
            - activation_func[Torch.nn]: Any activation function from torch as in https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Standard nn.Tanh

        Output:
            - torch object model. Linked Sequential layers

    '''
    def __init__(self, inputs, outputs, activation_func = torch.nn.Tanh):

        assert isinstance(inputs,int) and isinstance(outputs,int) and inputs > 0 and outputs > 0, "Input layer and Output layer must be non negative values except 0 !"
        
        # Inherit pytorch nn class
        super(ANN, self).__init__()

        self.inp = inputs
        self.out = outputs
        self.ann = [self.inp, 20, 20, 20, 20, 20, 20, 20, 20, self.out]
        self.depth = len(self.ann) - 1
        self.activation = activation_func

        layer_dict = {}
        for i in range(self.depth - 1): 
            layer_dict[f'layer_{i}'] = torch.nn.Linear(self.ann[i], self.ann[i+1])
            layer_dict[f'activation_{i}'] = self.activation()
        
        layer_dict[f'layer_{(self.depth - 1)}'] = torch.nn.Linear(self.ann[-2], self.ann[-1])
        
        # deploy layers
        self.layers = torch.nn.Sequential(OrderedDict(layer_dict)).to(device)


    # For prediciton 
    def forward(self, x):
        return self.layers(x)

# As in https://github.com/maziarraissi/PINNs
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(1.0)

class PhysicsInformedNN:
    # Initialize the class

    '''
        X is the matrix type object that contains the measurements of the input variables of the wanted function.
        For example: 
            given the wanted equation h(x,y,z,t) and its differential equation:
                        a_1*h_tt + a_2*h_xy + a_3*h_yy + a_4*h_yz = 0

        X will be of the following shape:
                                             0 , 1 , 2 , 3 
                        X[t, x, y, z] = [   [t1, x1, y1, z1 ],
                                            [t2, x2, y2, z2 ],
                                                   ...     
                                            [tn, xn, yn, zn ]]

        So to get y input values of the function h, just cut the matrix as X[:,2] => [y1, y2,..., yn]
                                                                           X[:,2:3] => [[y1, y2,..., yn]]
    '''
    def __init__(self, X, u, inputs, outputs, guess, lb, ub, SHOW_MODEL= False):
        
        
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device) # Lower bound
        self.ub = torch.tensor(ub).float().to(device) # Upper bound
        
        # data
        self.t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.u_in = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        
        # self.RL = torch.nn.Parameter(torch.tensor([random.uniform(-10.0,10.0)], requires_grad=True).to(device))
        # self.LC = torch.nn.Parameter(torch.tensor([random.uniform(-10.0,10.0)], requires_grad=True).to(device))
        self.RL = torch.nn.Parameter(torch.tensor([guess[0]], requires_grad=True).to(device))
        self.LC = torch.nn.Parameter(torch.tensor([guess[1]], requires_grad=True).to(device))
        
        # deep neural networks
        self.ann = ANN(inputs, outputs, activation_func = torch.nn.Tanh).to(device)
        self.ann.register_parameter('RL', self.RL)
        self.ann.register_parameter('LC', self.LC)
        self.best_params = self.ann.state_dict()
        self.ann.apply(init_weights)

        if SHOW_MODEL:
            print("\n\n---------------- MODEL ARCHITECTURE----------------\n")
            summary(self.ann,(1,inputs))
            print("\n\n")
        
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.ann.parameters(), 
            lr=1.0, 
            max_iter=5000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        
        self.optimizer_Adam = torch.optim.Adam(self.ann.parameters(), lr = 0.01)
        # self.optimizer_Adam = torch.optim.SGD(self.ann.parameters(), lr=0.1, momentum=0.9)
        self.iter = 0
        self.losses = []


    def net_u(self, t):  
        u = self.ann(torch.cat([t], dim=1))
        return u
    
    def net_f(self, t, u_in):
        """ The pytorch autograd version of calculating residual """
        RL = self.RL        
        LC = self.LC
        u = self.net_u(t)

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0]


        # u_tt = -( R/L )*u_t -( 1/(L*C) )*u + ( 1/(L*C) )*u_in
        f = u_tt + RL*u_t + LC*u - LC*u_in

        # I'm trying to understand why u(t) loss doesn't decrease for the moment.
        f.zero_()
        
        return f
    
    def loss_func(self):
        u_pred = self.net_u(self.t)
        f_pred = self.net_f(self.t, self.u_in)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 10 == 0:
            print(
                'Iter: %d, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e, R/L: %.3f, L*C: %.6f' % 
                (
                    self.iter,
                    loss.item(),
                    loss_u.item(),
                    loss_f.item(), 
                    self.RL.item(), 
                    self.LC.item()
                )
            )
        return loss
    
    def train(self, nIter, u_f = 10**0):

        # Setting the model in training mode
        self.ann.train()

        print(f'\n\n\t\tSTARTING ADAM !\n\n')
        lowest_loss = 999999
        for epoch in range(nIter):
            u_pred = self.net_u(self.t)
            f_pred = self.net_f(self.t, self.u_in)
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)

            # Calculating total loss
            loss = u_f*loss_u + loss_f

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            # loss.backward(gradient = torch.tensor([10.0, 1.0]) )
            loss.backward()
            self.optimizer_Adam.step()
            self.losses.append(loss.clone().detach().cpu().numpy())
            
            if epoch % 10 == 0:
                if (loss_f+loss_u) < lowest_loss:
                    torch.save(self.ann.state_dict(), './best-model-parameters.pt')
                    self.best_params = self.ann.state_dict()
                    lowest_loss = loss.clone()
                print(
                    'It: %d, Loss: %.3e, Loss_u: %.3e, Loss_f: %.3e, R/L: %.3f, L*C: %.6f' % 
                    (
                        epoch, 
                        loss.item(),
                        loss_u.item(), 
                        loss_f.item(), 
                        self.RL.item(), 
                        self.LC.item()
                    )
                )

        print(f'\n\n\t\tSTARTING FINE TUNE!\n\n')
        self.optimizer.step(self.loss_func)
    
    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        u_in = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.ann.load_state_dict(self.best_params)
        self.ann.eval()
        u = self.net_u(t)
        f = self.net_f(t, u_in)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


if __name__ == "__main__": 
    
    # ****************************** Initializating the model ******************************
    
    u_star = mat73.loadmat(r"D:\Documents\Lyon\INSA\Python\PINN_motor_model\u.mat")
    u_in_star = mat73.loadmat(r"D:\Documents\Lyon\INSA\Python\PINN_motor_model\sin.mat")

    X_star = u_in_star['sin'].T
    u_star = u_star['u'][1,:].T
    u_star = u_star[:,np.newaxis]

    R = 4
    L = 0.9
    C = 10*10**-6

    N_u = 8000
    iterations = 1000
    # guess = [np.random.normal(R/L, 0.1*np.sqrt(R/L), 1)[0],np.random.normal(L*C, 0.1*np.sqrt(L*C), 1)[0]]
    guess = [R/L+1, 1/(L*C)+1]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    print(f"Lower bounds: {lb}")
    print(f"Upper bounds: {ub}")

    fig= go.Figure(go.Scatter(x=X_star[:,0], y = X_star[:,1], mode ="lines", name="Input signal"))
    fig.add_scatter(x=X_star[:,0], y=u_star[:,0], mode='lines', name="Output signal")
    fig.write_html("./true_values.html")

    # ****************************** Training the model ******************************
    
    # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    model = PhysicsInformedNN(X_u_train, u_train, 1, 1, guess, lb, ub, SHOW_MODEL = True)

    print(f"\nInput variables: \n\tR: {R}\n\tL: {L}\n\tC: {C}\n")
    print(f"\nGuess R/L: {guess[0]}\t\t Guess L*C: {guess[1]}\n\n")
    model.train(iterations)

    # ****************************** Evaluating the model ******************************
    
    loss = list(model.losses)
    fig = go.Figure( go.Scatter(x=np.linspace(0,len(loss),len(loss)), y=loss ) )
    fig.write_html("./losses.html")  

    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    RL = model.RL.detach().cpu().numpy()
    LC = 1/model.LC.detach().cpu().numpy()

    error_lambda_1 = 100*np.abs(RL - R/L) / (R/L) 
    error_lambda_2 = 100*np.abs(LC - L*C) / (L*C) 

    print('\n\nError u: %e' % (error_u))
    print(f'R/L: {RL[0]}')
    print('Error R/L: %.5f%%' % (error_lambda_1[0]))
    print(f'L*C: {LC[0]}')
    print('Error L*C: %.5f%%' % (error_lambda_2[0]))

    fig= go.Figure(go.Scatter(x=X_star[:,0], y = u_star[:,0], mode ="lines", name="u_star"))
    fig.add_scatter(x=X_star[:,0], y=u_pred[:,0], mode='lines', name="u_pred")
    fig.write_html("./u_pred_vs_star.html")

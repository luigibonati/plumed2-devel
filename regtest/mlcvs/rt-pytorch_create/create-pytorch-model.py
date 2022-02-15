import torch

def my_torch_cv(x):
    '''
    Here goes the definition of the CV.

    Inputs:
        x (torch.tensor): input, either scalar or 1-D array
    Return:
        y (torch.tensor): collective variable (scalar)
    '''
    # CV definition
    y = torch.sin(x)

    return y

input_size = 1

# -- DEFINE INPUT -- 
#random 
#x = torch.rand(input_size, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#or by choosing the value(s) of the array
x = torch.tensor([0.], dtype=torch.float32, requires_grad=True)

# -- CALCULATE CV -- 
y = my_torch_cv(x)

# -- CALCULATE DERIVATIVES -- 
for yy in y:
    dy = torch.autograd.grad(yy, x, create_graph=True)
    # -- PRINT -- 
    with open('log', 'w') as f:
        print('Version:',torch.__version__,file=f)
        print('n_input\t: {}'.format(input_size),file=f)
        print('x\t: {}'.format(x),file=f)
        print('cv\t: {}'.format(yy),file=f)
        print('der\t: {}'.format(dy),file=f)

# Compile via tracing
traced_cv   = torch.jit.trace ( my_torch_cv, example_inputs=x )
# Compile via scripting
#scripted_cv = torch.jit.script( my_torch_cv )

filename='torch_model.pt'
traced_cv.save(filename)

# -- SAVE SERIALIZED FUNCTION -- 
#filename='torch_cv_scripted.pt'
#scripted_cv.save(filename)
 

"""Tools for plotting diagnostic info."""

from matplotlib import pyplot as plt
import numpy as np
figsize = (8,6)

def plot_trig_poly_magnitude(p, ax=None, points=200, c='blue'):
    ax = ax or plt.gca()

    ts = np.linspace(0.0, 1.0, points)
    values = p(ts)
    if len(values.shape) == 1:
        ys = np.absolute(values)
    else:
        ys = np.linalg.norm(values, axis=0)

    ax.plot(ts, ys, c=c)
    plt.ylabel('N(t)')
    
def plot_trig_poly_magnitude_der_2ndder(p, support, ax=None, points=200, c='blue'):    
    ts = np.linspace(0.0, 1.0, points)
    values = p(ts)
    values_1 = p.derivative(ts)
    values_2 = p.derivative2(ts)
    N_1 = 2*np.real(np.diagonal(np.matrix(values_1).H.dot(values) ))
    N_2 = 2*np.real(np.diagonal(np.matrix(values_2).H.dot(values) )) + 2*(np.diagonal(np.matrix(values_1).H.dot(values_1) ))
    
    plt.figure()
    ax = ax or plt.gca()
    ax.plot(ts, N_1, c=c)
    ax.plot(support, np.zeros(len(support)), 'go', ms = 5)
    plt.ylabel('N\'(t)')
    print(2*np.real(np.diagonal(np.matrix(p.derivative(support)).H.dot(p(support)) )))
        
    plt.figure()
    ax = plt.gca()
    ax.plot(ts, N_2, c=c)
    ax.plot(support, np.zeros(len(support)), 'go', ms = 5)
    plt.ylabel('N\'\'(t)')
    print(2*np.real(np.diagonal(np.matrix(p.derivative2(support)).H.dot(p(support)) )) + 2*(np.diagonal(np.matrix(p.derivative(support)).H.dot(p.derivative(support)) )))

def plot_trig_poly_each_interpolant(p, support, ax=None, points=200, c='blue'):
    ts = np.linspace(0.0, 1.0, points)
    values = p(ts)    
    vshape = values.shape 
    if len(vshape) == 1:
        plt.figure(0)
        ax = ax or plt.gca()
        ys = np.absolute(values)
        ax.plot(ts, ys, c=c)
    else:
        for i in range(vshape[0]):
            plt.figure(i)
            ax = plt.gca()
            ys = values[i,:]
            ax.plot(ts, np.absolute(ys), c=c)
            plot_support_magnitude_lines(support)
            plot_magnitude_bounds()

            
            
def plot_individual_magnitude(p, support, qs, ts, points=200, plot_Nt = True):
    values = p(ts)
    
    plt.figure(figsize=figsize, dpi=100)
    ax = plt.gca()
    ys = np.linalg.norm(values, axis=0)
    if plot_Nt:
        ax.plot(ts, ys, c='C0', label = r'$N(t)$')
        
    j=1
    
    for k in [qss-1 for qss in qs]:
        ys = values[k,:]
        ax.plot(ts, np.absolute(ys), c = 'C'+str(j), label = r'$\Vert q_{%d}(t)\Vert $'%(k+1) )
        j  = j+1
        
    plot_support_magnitude_lines(support[[qss-1 for qss in qs]], start = 0.9,  c = 'C'+str(j) )
    j = j+1
    plot_magnitude_bounds(ts[0], ts[-1],  c = 'C'+str(j))
    
    leg = plt.legend(loc='best',ncol=3, fancybox=True)
    leg.get_frame().set_alpha(0.5)

def plot_individual_components(coeffs, support, kernel, kernel_1, qs, ts, hops, diff_color = False):

    plt.figure(figsize=figsize, dpi=100)
    ax = plt.gca()
    j = 1;
    n = support.shape[0]
    for k in [qss-1 for qss in qs]:
        for s in range(max( k - hops, 0), min(k+hops+1, n)  ):
            alp_k_s = kernel.sum_shifts([-support[s]], [coeffs[4*k*n+s]] ) + kernel.sum_shifts([-support[s]], [coeffs[(4*k+2)*n+s] * 1j])
            ax.plot(ts, np.real(alp_k_s(ts)), c = 'C'+str(j), label=r'Re{$\alpha_{%d,%d} K(t-t_{%d})} $'%(s+1,k+1,s+1) )
            if diff_color:
                j = j+1
            if s == k:
                beta_k_s = kernel_1.sum_shifts([-support[s]], [coeffs[(4*k+1)*n+s]] ) + kernel_1.sum_shifts([-support[s]], [coeffs[(4*k+3)*n + s] * 1j] )            
                ax.plot(ts, np.real(beta_k_s(ts)), '--', c = 'C'+str(j), label = r'Re{$\beta_{%d} K^\prime(t-t_{%d})$}'%(k+1,k+1) )
                if diff_color:
                    j = j+1
        if ~diff_color:
            j  = j+1
    
    plot_support_magnitude_lines(support[[qss-1 for qss in qs]], start = 0.9,  c = 'C'+str(j) )
    j = j+1
    plot_magnitude_bounds(ts[0], ts[-1],  c = 'C'+str(j))

    leg = plt.legend(loc='best',ncol=2, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
def plot_2ndder_zoom(p, support, q, ts, hops):    
    values = p(ts)
    values_1 = p.derivative(ts)
    values_2 = p.derivative2(ts)
    n = values_1.shape[0]
    
    plt.figure(figsize=figsize, dpi=100)
    ax = plt.gca()
    j = 1
    y_1 = 2*np.real(np.diagonal(np.matrix(values).H.dot(np.matrix(values_2)) ))
    ax.plot(ts, y_1, c = 'C'+str(j), label = r'2Re{$\langle q(t), q^{\prime\prime}(t)\rangle $}' )
    j = j+1
    N_2 = y_1
    for i in range(max( q - hops, 0), min(q + hops+1, n)  ):
        if i == q:
            continue
            # j = j+1
        else:            
            y_2_i = 2*(np.diagonal(np.matrix(values_1[i,:]).H.dot(np.matrix(values_1[i,:]) ) ))
            ax.plot(ts, y_2_i, c = 'C'+str(j), label = r'$2\Vert q_{%d}(t)\Vert ^2$'%(i))
            N_2 = N_2 + y_2_i
            j = j+1

    ax.plot(ts, N_2, c = 'C'+str(j), label = r'$N^{\prime\prime}(t)$')
    j = j+1
    
    ax.plot(support[q-1], 0, 'C'+str(j)+'o', ms = 5) 
    j = j+1
    ax.plot([support[i] for i in range(len(support)) if i!=q-1 ], np.zeros(len(support)-1), 'C'+str(j)+'o', ms = 4) 
    
    j = j+1
    plot_magnitude_bounds(ts[0], ts[-1],  c = 'C'+str(j))
    
    leg = plt.legend(loc='best',ncol=3, fancybox=True, mode= 'expand')
    leg.get_frame().set_alpha(0.5)
    
    
def plot_support_magnitude_lines(support, start= 0.0, height=1.0, ax=None, c='green'):
    ax = ax or plt.gca()

    ax.vlines(support, start, height, color=c)


def plot_magnitude_bounds(xmin=0.0, xmax=1.0, c='red'):
    ax = plt.gca()

    ax.hlines([1.0], xmin, xmax, color=c)

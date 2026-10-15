import numpy as np
from evoscape.modules import Node, UnstableNode, Center
from evoscape.landscapes import Landscape
from evoscape.morphogen_regimes import mr_const



def update_landscape_umbilic(p1, p2):
    sig=0.52
    s_head = sig * np.exp(+p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    x_fate = 1.0
    y_cycle = 0.5
    x_cycle = 0.
    modules = [
        Node(x=x_cycle, y=y_cycle, a=4., s=s_head),
        Node(x=-x_fate, y=-1., a=4.50, s=s_left+0.01),
        Node(x=+x_fate, y=-1., a=4.5, s=s_right),
    ]
    return Landscape(modules, A0=0.05, x0=(0., -0.8), regime=mr_const, n_regimes=1)


def update_landscape_umbilic_cycle(p1, p2):
    sig=0.52
    s_head = sig * np.exp(+p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    x_fate = 1.0
    y_cycle = 0.5
    x_cycle = 0.
    modules = [
        Node(x=x_cycle, y=y_cycle, a=4., s=s_head),
        Node(x=-x_fate, y=-1., a=4.51, s=s_left+0.01),
        Node(x=+x_fate, y=-1., a=4.5, s=s_right),

        UnstableNode(x=x_cycle, y=y_cycle, a=6., s=s_head*0.8),
        Center(x=x_cycle, y=y_cycle, a=4., s=s_head*0.85),
    ]
    return Landscape(modules, A0=0.05, x0=(0., -0.8), regime=mr_const, n_regimes=1)


def update_landscape_binary_choice(p1, p2):
    sig=0.5
    s_head = sig * np.exp(+p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    x_fate = 1.8 #1.8
    modules = [
        Node(x=-0.0, y=0., a=4., s=s_head),
        Node(x=-x_fate, y=0., a=4.51, s=s_left),
        Node(x=+x_fate, y=0., a=4.5, s=s_right),
    ]
    return Landscape(modules, A0=0.05, x0=(0., 0.), regime=mr_const, n_regimes=1)


def update_landscape_heteroclinic(p1, p2):
    sig=0.5
    s_head = sig * np.exp(+p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    x_fate = 1.8 #1.8
    modules = [
        Node(x=-0.0, y=0., a=4., s=s_head),
        Node(x=-x_fate, y=0., a=4.51, s=s_left),
        Node(x=+x_fate, y=0., a=4.5, s=s_right),
        UnstableNode(x=0.0, y=0, a=6., s=s_head*0.8),
        Center(x=0, y=0., a=4., s=s_head*0.85),
    ]
    return Landscape(modules, A0=0.05, x0=(0., 0.), regime=mr_const, n_regimes=1)


def update_landscape_double_snic(p1, p2):
    sig=0.5
    s_head = sig * np.exp(+p2 / 2)
    s_fate = sig * np.exp(-p2 / 2)
    s_left = s_fate * np.exp(-p1 / 2)
    s_right = s_fate * np.exp(p1 / 2)

    x_fate = 1.1
    modules = [
        Node(x=-0.0, y=0., a=4., s=s_head),
        Node(x=-x_fate, y=0., a=4.51, s=s_left),
        Node(x=+x_fate, y=0., a=4.5, s=s_right),
        UnstableNode(x=0.0, y=0, a=6., s=s_head*0.8), #
        Center(x=0, y=0., a=4., s=s_head*0.85),       #
    ]
    return Landscape(modules, A0=0.05, x0=(0., 0.), regime=mr_const, n_regimes=1)





# L = 2
# npoints = 401
# q = np.linspace(-L, L, npoints)
# xx, yy = np.meshgrid(q,q, indexing = 'xy')
#
# x_fate = 1.0 #1.1, 1.5
# y_cycle = 0.8 # 0.5
# y_fate = -1
# tau = 2
# # M1 a = 4 and M4 a = 2 (without R and C)
# M1 = Node(x=-0.0, y=y_cycle, a=6., s=(0.5, 0, 1), tau=tau)
# M2 = Node(x=-x_fate, y=y_fate, a=5., s=(0.5, -1, -1), tau=tau)
# M3 = Node(x=+x_fate, y=y_fate, a=5., s=(0.5, +1, -1), tau=tau)
#
# M4 = Node(x=0., y=y_fate, a=2., s=(1.5*0.5, 0, -1), tau=tau)
#
# R = UnstableNode(x=0.0, y=y_cycle, a=8., s=(0.8*0.5, 0, 1), tau=tau)
# C= Center(x=0, y=y_cycle, a=4., s=(0.85*0.5, 0, 1), tau=tau)
# C2= Center(x=0, y=y_fate*0.5, a=1., s=(0.5, 0, 1), tau=tau)
#
# modules = [M1, M2, M3, M4, R, C]
# land = Somitogenesis_Landscape(modules, A0=0.05, x0=(0., -0.8), regime=two_controls, n_regimes=3)



# L = 2
# npoints = 401
# q = np.linspace(-L, L, npoints)
# xx, yy = np.meshgrid(q,q, indexing = 'xy')
#
# x_fate = 1.0 #1.1, 1.5
# y_cycle = 0.8 # 0.5
# y_fate = -1
# tau = 2
# # M1 a = 4 and M4 a = 2 (without R and C)
# M1 = Node(x=-0.0, y=y_cycle, a=6., s=(0.5, 0, 1), tau=tau)
# M2 = Node(x=-x_fate, y=y_fate, a=5., s=(0.5, -1, -1), tau=tau)
# M3 = Node(x=+x_fate, y=y_fate, a=5., s=(0.5, +1, -1), tau=tau)
#
# M4 = Node(x=0., y=y_fate, a=2., s=(1.5*0.5, 0, -1), tau=tau)
# # M5 = UnstableNode(x=0., y=y_fate, a=5., s=(0.5*0.5, 0, -1.), tau=tau)
#
# # R = UnstableNode(x=0.0, y=y_cycle, a=8., s=(0.8*0.5, 0, 1), tau=tau)
# # C= Center(x=0, y=y_cycle, a=4., s=(0.75*0.5, 0, 1), tau=tau)
# # C2= Center(x=0, y=y_fate*0.5, a=1., s=(0.5, 0, 1), tau=tau)
#
# modules = [M1, M2, M3, M4]
# land = Somitogenesis_Landscape(modules, A0=0.05, x0=(0., -1.), regime=two_controls, n_regimes=3)



#     # binary choice / snic
#
# L = 2
# npoints = 401
# q = np.linspace(-L, L, npoints)
# xx, yy = np.meshgrid(q,q, indexing = 'xy')
#
# x_fate = 1.8
# y_cycle = 0.
# y_fate = 0.
# sig=0.5 # 0.6
# tau = 5.
#
#
# modules = [
#     Node(x=-0.0, y=y_cycle, a=4., s=(sig, 0, +1), tau=tau),
#
#     Node(x=-x_fate, y=y_fate, a=4.51, s=(sig, -1, -1), tau=tau),
#     Node(x=+x_fate, y=y_fate, a=4.5, s=(sig, +1, -1), tau=tau), # a = 5
#
#     # Node(x=0.0, y=y_fate, a=3., s=(1.*0.5, 0, +1), tau=tau),
#
#     UnstableNode(x=-0.0, y=y_cycle, a=6., s=(sig*0.8, 0, +1), tau=tau),
#     Center(x=0, y=y_cycle, a=4., s=(0.85*sig, 0, +1), tau=tau)
#
# ]
# land = Somitogenesis_Landscape(modules, A0=0.05, x0=(0., 0.), regime=two_controls, n_regimes=3)




# simple flip (flip with cusp)

# L = 2
# npoints = 401
# q = np.linspace(-L, L, npoints)
# xx, yy = np.meshgrid(q,q, indexing = 'xy')
#
# x_fate = 1.0 #1.1, 1.5
# y_cycle = 0.8 # 0.5
# y_fate = -1
# tau = 2
# # M1 a = 4 and M4 a = 2 (without R and C)
# M1 = Node(x=-0.0, y=y_cycle, a=5., s=(0.5, 0, 1), tau=tau)
# M2 = Node(x=-x_fate, y=y_fate, a=5.1, s=(0.5, -1, -1), tau=tau)
# M3 = Node(x=+x_fate, y=y_fate, a=5., s=(0.5, +1, -1), tau=tau)
#
# M4 = Node(x=0., y=y_fate-0.1, a=3., s=(1.0*0.5, 0, 1), tau=tau)
# # M5 = UnstableNode(x=0., y=y_fate, a=5., s=(0.5*0.5, 0, -1.), tau=tau)
#
# # R = UnstableNode(x=0.0, y=y_cycle, a=8., s=(0.8*0.5, 0, 1), tau=tau)
# # C= Center(x=0, y=y_cycle, a=4., s=(0.75*0.5, 0, 1), tau=tau)
# # C2= Center(x=0, y=y_fate*0.5, a=1., s=(0.5, 0, 1), tau=tau)
#
# modules = [M1, M2, M3, M4]
# land = Somitogenesis_Landscape(modules, A0=0.1, x0=(0., -1.), regime=two_controls, n_regimes=3)




# # flip with cusp by increasing size
#
# L = 2
# npoints = 401
# q = np.linspace(-L, L, npoints)
# xx, yy = np.meshgrid(q,q, indexing = 'xy')
#
# x_fate = 0.8
# y_cycle = 0.8
# y_fate = -1.
# sig=0.5 # 0.6
# tau = 2.
#
#
# modules = [
#     Node(x=-0.0, y=y_cycle, a=6., s=(sig, 0, +1), tau=tau),
#     Node(x=-x_fate, y=y_fate, a=4.05, s=(sig, -1, +1), tau=tau),
#     Node(x=+x_fate, y=y_fate, a=4., s=(sig, +1, +1), tau=tau), # a = 5
#     Node(x=0.0, y=y_fate, a=1., s=(1.5*0.5, 0, -1), tau=tau),
#
#     UnstableNode(x=-0.0, y=y_cycle, a=7., s=(sig*0.8, 0, +1), tau=tau),
#     Center(x=0, y=y_cycle, a=4., s=(0.85*sig, 0, +1), tau=tau)
#
# ]
# land = Somitogenesis_Landscape(modules, A0=0.05, x0=(0., -1.), regime=two_controls, n_regimes=3)
import numpy as np


def H_b(z):
    h_b = -(np.multiply(z, np.log2(z + 1e-10)) + np.multiply(1 - z, np.log2(1 - z + 1e-10)))

    return h_b

# Parent Class for a general unifilar channel.
# That is p(y,s'|s,x) = p(y|s,x)1[s'=f(x,s,y)]
class Unifilar(object):
    def __init__(self, size, P_out, P_state, state_dim=1, action_dim=2):
        self._size = size

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._P_out = P_out
        self._P_state = P_state
        self._eps = 1e-20
        self.z = None


    def _expand_u(self, a):
        u = np.reshape(a, [self._size, 1, self._action_dim])
        return np.concatenate([u, 1 - u], axis=1)

    def _expand_z(self):
        return np.tile(np.concatenate([self.z, 1 - self.z], axis=2), [1, 2, 1])

    def _reduce_z(self, z):
        return np.reshape(z[:, 0], [self._size, 1, self._state_dim])

    def step(self, a):
        if np.max(a) > 1 or np.min(a) < 0:
            raise ValueError("Environment got invalid action")
        if self.z is None:
            raise ValueError("DP state is not initialized")

        u = self._expand_u(a)
        z = self._expand_z()

        r = self._reward(z,u)
        z_prime = self._next_state(z, u)

        self.z = z_prime

        return self.z, r

    def _reward(self,z,u):
        y_cardinal = self._P_out.shape[-1]
        p_xsy = np.zeros([self._size,2,2,y_cardinal])

        for i in range(y_cardinal):
            p_xsy[:, :, :, i] = z * u * np.expand_dims(self._P_out[:, :, i], axis=0)

        self._p_y = p_y = np.sum(p_xsy, axis=(1,2)) / np.reshape(np.sum(p_xsy + self._eps, axis=(1,2,3)), [self._size,1])

        r = np.sum(-p_y * np.log2(p_y+self._eps), axis=1) - \
            np.sum(-p_xsy * np.log2(self._P_out+self._eps), axis=(1,2,3))

        return np.reshape(r, [self._size, 1])

    def _next_state(self, z,u):
        y_cardinal = self._P_out.shape[-1]

        def f(z,u,p_o,p_s):
            p = np.zeros([self._size, 2, 2, 2])
            p[:, :, :, 0] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 0], axis=0)
            p[:, :, :, 1] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 1], axis=0)
            return np.sum(p, axis=(1,2)) / np.reshape(np.sum(p, axis=(1,2,3)) + self._eps, [self._size,1])

        w = np.random.binomial(1, np.reshape(self._p_y[:, 1], [-1,1]), size=[self._size, 1])

        # p_cum = np.cumsum(self._p_y, axis=1)
        # noise = np.tile(np.random.rand(self._size, 1), [1, y_cardinal])
        # noise2 = (noise < p_cum) * 1
        # w = np.reshape(np.argmax(noise2, axis=1), [self._size, 1])

        z_prime = np.where(w == 0, f(z,u,self._P_out[:, :, 0],self._P_state[:, :, 0, :]),
                           f(z, u, self._P_out[:, :, 1], self._P_state[:, :, 1, :]))

        return self._reduce_z(z_prime)

    def reset(self):
        self.z = np.random.rand(self._size, 1, self._state_dim)
        return self.z

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def size(self):
        return self._size

    @staticmethod
    def optimal_bellman(N):
        h_ast = np.ones(N)

        a_ast_0 = np.ones(N)
        a_ast_1 = np.ones(N)
        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast



class Trapdoor(Unifilar):
    state_dim = 1
    action_dim = 2

    state_cardin = 2
    input_cardin = 2
    output_cardin = 2

    def __init__(self, size):
        def f(x, s, y):
            return np.logical_xor(x, np.logical_xor(s, y)) * 1

        P_out = np.array([[[1,0],[0.5,0.5]],[[0.5, 0.5],[0,1]]])
        P_state = np.array([[[[(s_prime == f(x,s,y)) * 1    for s_prime in range(self.state_cardin)]
                                                            for y in range(self.output_cardin)]
                                                            for s in range(self.state_cardin)]
                                                            for x in range(self.input_cardin)])
        super().__init__(size, P_out, P_state, self.state_dim, self.action_dim)

    @staticmethod
    def optimal_bellman(N):
        s_vec = np.linspace(0, 1, N)
        z_0 = 0.0
        z_1 = 0.382
        z_2 = 0.613
        z_3 = 1.0
        ro_ast = 0.694241

        # optimal state-value function
        h_ast = np.zeros(N)
        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            H_b(s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]) - \
            ro_ast * s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)] + np.log2(np.sqrt(5)-1)
        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 1.0
        h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)] =  \
            H_b(s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]) + \
            ro_ast * s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)] + np.log2(3-np.sqrt(5))

        # optimal policy
        a_ast_0 = np.zeros(N)
        a_ast_0[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]
        a_ast_0[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_0[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]

        a_ast_1 = np.zeros(N)
        a_ast_1[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                (1 - s_vec[
                                                                    np.logical_and(s_vec >= z_0, s_vec <= z_1)])
        a_ast_1[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_1[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = (
                    1 - s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)])

        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast

class Ising(Unifilar):
    state_dim = 1
    action_dim = 2
    def __init__(self, size):
        def f(a, b, c):
            return a
        P_out = np.array([[[1, 0], [0.5, 0.5]], [[0.5, 0.5], [0, 1]]])
        P_state = np.array([[[[(s_prime == f(x,s,y)) * 1    for s_prime in range(2)]
                                                            for y in range(2)]
                                                            for s in range(2)]
                                                            for x in range(2)])
        super().__init__(size, P_out, P_state, self.state_dim, self.action_dim)

    @staticmethod
    def optimal_bellman(N):
        s_vec = np.linspace(0, 1, N)
        c = 0.4503
        z_0 = 0.0
        z_1 = (1-c)/(1+c)
        z_2 = (2*c)/(1+c)
        z_3 = 1.0
        ro_ast = 0.575522

        # optimal state-value function
        h_ast = np.zeros(N)
        s = s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]
        h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = \
            1/(1-c) * H_b((2*c+(1-c)*s)/2) - \
            s + (c*s-4*c-s)/(2-2*c) * ro_ast + \
            (2*c + (1-c)*s) / (2 - 2*c) * H_b((2*c)/(c*(2-s)+s))

        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = \
            H_b(s_vec[np.logical_and(s_vec >= z_1, s_vec <= z_2)])

        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] =  h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)][::-1]

        # optimal policy
        a_ast_0 = np.zeros(N)
        a_ast_0[np.logical_and(s_vec >=z_0, s_vec <= z_2)] = \
            s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_2)]
        a_ast_0[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = \
            c * (2 - s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)])

        a_ast_1 = np.zeros(N)
        a_ast_1[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            c * (1 + s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)])
        a_ast_1[np.logical_and(s_vec >= z_1, s_vec <= z_3)] = \
            1 - s_vec[np.logical_and(s_vec >= z_1, s_vec <= z_3)]
        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast

class Bec_nc1(Unifilar):
    state_dim = 1
    action_dim = 1
    def __init__(self, size, e):
        def f(a, b, c):
            return a
        P_out = np.array([[[1-e, e, 0], [1-e, e, 0]], [[0, e, 1-e], [0, e, 1-e]]])
        P_state = np.array([[[[(s_prime == f(x,s,y)) * 1    for s_prime in range(2)]
                                                            for y in range(3)]
                                                            for s in range(2)]
                                                            for x in range(2)])
        super().__init__(size, P_out, P_state, self.state_dim, self.action_dim)
        self.e = e

    def _next_state(self, z,u):
        y_cardinal = self._P_out.shape[-1]

        def f(z,u,p_o,p_s):
            p = np.zeros([self._size, 2, 2, 2])
            p[:, :, :, 0] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 0], axis=0)
            p[:, :, :, 1] = z * u * np.expand_dims(p_o, axis=0) * np.expand_dims(p_s[:, :, 1], axis=0)
            return np.sum(p, axis=(1,2)) / np.reshape(np.sum(p, axis=(1,2,3)) + self._eps, [self._size,1])

        assert(np.logical_and(self._p_y > 0.0, self._p_y < 1.0 ), "invalid diturbance distribution")
        # w = np.reshape(np.array([np.random.choice(y_cardinal, 1, p=self._p_y[i,:]) for i in range(self._size)]), [self._size, 1])

        p_cum = np.cumsum(self._p_y, axis=1)
        noise = np.tile(np.random.rand(self._size, 1), [1, y_cardinal])
        noise2 = (noise < p_cum)*1
        w = np.reshape(np.argmax(noise2, axis=1), [self._size, 1])

        z_prime = np.where(w == 0, f(z,u,self._P_out[:, :, 0],self._P_state[:, :, 0, :]),
                           np.where(w == 1, f(z, u, self._P_out[:, :, 1], self._P_state[:, :, 1, :]),
                                            f(z, u, self._P_out[:, :, 2], self._P_state[:, :, 2, :])))
        return self._reduce_z(z_prime)

    def _expand_u(self, a):
        u = np.reshape(a, [self._size, 1, self._action_dim])
        u = np.concatenate([u, 1-u], axis=1)
        u_1 = np.concatenate([np.ones([self._size, 1, 1]), np.zeros([self._size, 1, 1])], axis=1)
        return np.concatenate([u, u_1], axis=2)

    @staticmethod
    def capacity(eps):
        p = np.linspace(0.0001, 0.5, 10000)

        h_b = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        ro = np.max(h_b / (p + 1/(1-eps)))
        k = np.argmax(h_b / (p + 1/(1-eps)))
        p_e = p[k]
        return ro, p_e

    def optimal_bellman(self, N):
        s_vec = np.linspace(0, 1, N)
        e = self.e
        ro_ast, p_ast = self.capacity(e)

        z_0 = 0.0
        z_1 = p_ast
        z_2 = 1.0

        # optimal state-value function
        h_ast = np.zeros(N)
        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = ro_ast
        z = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]
        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            (1-e) * H_b(z) - z * (1-e) * ro_ast

        # optimal policy
        z = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]

        a_ast = np.zeros(N)
        a_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = z
        a_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = p_ast
        a_ast = np.reshape(a_ast, [1, N])
        return h_ast, a_ast



class BEC121(Unifilar):
    state_dim = 2
    action_dim = 4

    state_cardin = 2
    input_cardin = 2
    output_cardin = 6

    def __init__(self, size, e):
        def multivar_ind(x, y):
            return (np.mean((x ==  y) * 1) == 1) * 1

        def f_s(x, s, y):
            s = np.unpackbits(np.array([s], dtype=np.uint8))[-2:]
            return np.array((x,s[0]))

        def f_y(x,s,y):
            if y == 5:
                return e
            else:
                s = np.unpackbits(np.array([s], dtype=np.uint8))[-2:]
                return multivar_ind(y, x + 2*s[0] + s[1]) * (1-e)

        P_out = np.array([[[f_y(x,s,y)   for y in range(self.output_cardin)]
                                         for s in range(self.state_cardin ** self.state_dim)]
                                         for x in range(self.input_cardin)])

        P_state = np.array([[[[multivar_ind(s_prime, f_s(x,s,y))    for s_prime in range(self.state_cardin ** self.state_dim)]
                                                                    for y in range(self.output_cardin)]
                                                                    for s in range(self.state_cardin ** self.state_dim)]
                                                                    for x in range(self.input_cardin)])

        super().__init__(size, P_out, P_state, self.state_dim, self.action_dim)

    @staticmethod
    def optimal_bellman(N):
        s_vec = np.linspace(0, 1, N)
        z_0 = 0.0
        z_1 = 0.382
        z_2 = 0.613
        z_3 = 1.0
        ro_ast = 0.694241

        # optimal state-value function
        h_ast = np.zeros(N)
        h_ast[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = \
            H_b(s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]) - \
            ro_ast * s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)] + np.log2(np.sqrt(5)-1)
        h_ast[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 1.0
        h_ast[np.logical_and(s_vec >= z_2, s_vec <= z_3)] =  \
            H_b(s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]) + \
            ro_ast * s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)] + np.log2(3-np.sqrt(5))

        # optimal policy
        a_ast_0 = np.zeros(N)
        a_ast_0[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = s_vec[np.logical_and(s_vec >= z_0, s_vec <= z_1)]
        a_ast_0[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_0[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)]

        a_ast_1 = np.zeros(N)
        a_ast_1[np.logical_and(s_vec >= z_0, s_vec <= z_1)] = 0.5 * (np.sqrt(5) - 1) * \
                                                                (1 - s_vec[
                                                                    np.logical_and(s_vec >= z_0, s_vec <= z_1)])
        a_ast_1[np.logical_and(s_vec >= z_1, s_vec <= z_2)] = 0.5 * (3 - np.sqrt(5))
        a_ast_1[np.logical_and(s_vec >= z_2, s_vec <= z_3)] = (
                    1 - s_vec[np.logical_and(s_vec >= z_2, s_vec <= z_3)])

        a_ast = np.stack((a_ast_0, a_ast_1), axis=0)

        return h_ast, a_ast
